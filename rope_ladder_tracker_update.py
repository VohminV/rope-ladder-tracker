#!/usr/bin/env python3
"""
Улучшенная версия Rope Ladder Tracker (один файл).
Оптимизации:
 - потоковый захват кадров
 - повторное использование CLAHE и параметров
 - опциональная UMat/OpenCL поддержка (если OpenCV собран)
 - RANSAC-оценка affine для удаления выбросов
 - простая Kalman фильтрация положения
"""

import cv2
import numpy as np
import time
import logging
import json
import os
import threading

# ----------------- Настройки (под Orange Pi 5) -----------------
IMAGE_WIDTH_PX = 640   # можно снизить до 320 для большей производительности
IMAGE_HEIGHT_PX = 480
TARGET_FPS = 20        # realistic for ARM device; регулируй
FRAME_INTERVAL = 1.0 / TARGET_FPS

MIN_FEATURES = 20
MAX_FEATURES = 600
DISTANCE_THRESHOLD = 28.0
BACKTRACK_MARGIN = 14.0
HYSTERESIS_MARGIN = 10.0
LADDER_UPDATE_INTERVAL = 0.5
SMOOTHING_FACTOR = 0.75  # EMA для центра (доп. к Kalman)
FLAG_PATH = 'tracking_enabled.flag'
OFFSETS_FILE = 'offsets.json'

# ROI: None или (x,y,w,h) — если знаешь область интереса, укажи, чтобы ускорить
ROI = None  # e.g. (100,50,400,300)

# --- Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rope_ladder_improved.log", mode='w', encoding='utf-8')
    ]
)

# ----------------- Сохранение offsets -----------------
def save_offset(dx_m, dy_m, angle=0.0):
    data = {
        'x': float(dx_m),
        'y': float(dy_m),
        'angle': float(angle)
    }
    tmp = OFFSETS_FILE + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, OFFSETS_FILE)
    except Exception as e:
        logging.warning(f"Не удалось сохранить offsets: {e}")

# ----------------- Проверка флага трекинга -----------------
def is_tracking_enabled():
    try:
        with open(FLAG_PATH, 'r') as f:
            return f.read().strip() == '1'
    except:
        return False

# ----------------- Поток захвата кадров -----------------
class FrameGrabber(threading.Thread):
    def __init__(self, src=0, width=IMAGE_WIDTH_PX, height=IMAGE_HEIGHT_PX, fps=TARGET_FPS):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.lock = threading.Lock()
        self.running = True
        self.latest_frame = None
        self._ready = threading.Event()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest_frame = frame
                self._ready.set()
        self.cap.release()

    def read(self, timeout=1.0):
        # wait for a frame to be ready
        ok = self._ready.wait(timeout)
        if not ok:
            return False, None
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            self._ready.clear()
        return frame is not None, frame

    def stop(self):
        self.running = False

# ----------------- Простой Kalman фильтр для центра -----------------
class SimpleKalman2D:
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        # state [x, y, vx, vy]
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ],
        ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.initialized = False

    def correct_and_predict(self, meas):
        meas = np.array(meas, dtype=np.float32).reshape(2, 1)
        if not self.initialized:
            # init state
            self.kf.statePost = np.array([[meas[0,0]], [meas[1,0]], [0.0], [0.0]], dtype=np.float32)
            self.initialized = True
            return np.array([meas[0,0], meas[1,0]])
        self.kf.correct(meas)
        pred = self.kf.predict()
        return np.array([pred[0,0], pred[1,0]])

# ----------------- Утилиты для UMat -----------------
def to_umat_if_available(img):
    # используем UMat если доступно — ускорение при сборке OpenCV с OpenCL
    if hasattr(cv2, 'UMat'):
        try:
            return cv2.UMat(img)
        except:
            return img
    return img

def from_umat(x):
    if hasattr(x, 'get'):
        try:
            return x.get()
        except:
            return x
    return x

# ----------------- Фабрика адаптивного детектора точек -----------------
class FeatureDetector:
    def __init__(self, min_features=MIN_FEATURES, max_features=MAX_FEATURES):
        self.min_features = min_features
        self.max_features = max_features
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # предварительные параметры, можно подбирать
        self.quality_level = 0.03
        self.min_distance = 12
        self.block_size = 7

    def detect(self, gray):
        if gray is None:
            return None
        # в UMat-режиме требуется get для numpy в некоторых функциях -> применять осторожно
        # Применим CLAHE и blur
        try:
            if hasattr(gray, 'get'):
                g = from_umat(gray)
            else:
                g = gray
            enhanced = self.clahe.apply(g)
            blurred = cv2.GaussianBlur(enhanced, (3,3), 0)
            h, w = blurred.shape
            area = h*w
            # адаптивное maxCorners в зависимости от размера
            num = max(self.min_features, min(self.max_features, int(area / 600)))
            pts = cv2.goodFeaturesToTrack(
                image=blurred,
                maxCorners=num,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                useHarrisDetector=False
            )
            if pts is None or len(pts) < self.min_features:
                # ослабляем требования: уменьшение qualityLevel и minDistance
                pts = cv2.goodFeaturesToTrack(
                    image=blurred,
                    maxCorners=max(self.min_features, num),
                    qualityLevel=max(0.005, self.quality_level * 0.5),
                    minDistance=max(3, int(self.min_distance/2)),
                    blockSize=5,
                    useHarrisDetector=False
                )
            if pts is None:
                return None
            pts = pts.reshape(-1, 2)
            return pts
        except Exception as e:
            logging.debug(f"[FeatureDetector] detect failed: {e}")
            return None

# ----------------- Rope ladder waypoint management (улучшено) -----------------
def add_waypoint(waypoints, pts, angle=None, frame_idx=None, gray=None):
    if pts is None or len(pts) < MIN_FEATURES:
        return
    wp = {
        'frame': frame_idx,
        'points': pts.copy(),
        'angle': angle,
        'center': np.mean(pts, axis=0),
        'gray': gray  # storing reference (careful with UMat)
    }
    waypoints.append(wp)
    logging.debug(f"Добавлена WP (total={len(waypoints)})")

def rope_ladder_waypoint_management(waypoints, current_pts, distance_threshold=DISTANCE_THRESHOLD,
                                   anchor_center_fixed=None, frame_gray=None):
    """
    Добавление/удаление точек по логике 'верёвочной лестницы'.
    Использует расстояния между центрами waypoints и текущим центром.
    """
    if len(waypoints) == 0:
        return waypoints

    if current_pts is None or len(current_pts) == 0:
        return waypoints

    curr_center = np.mean(current_pts, axis=0)
    anchor_center = anchor_center_fixed if anchor_center_fixed is not None else waypoints[0]['center']
    current_to_anchor_dist = np.linalg.norm(curr_center - anchor_center)

    # найдем ближайшую wp
    dists = np.array([np.linalg.norm(wp['center'] - curr_center) for wp in waypoints])
    closest_idx = int(np.argmin(dists))
    closest_dist = float(dists[closest_idx])

    # добавление: если текущая позиция явно дальше последней (на BACKTRACK_MARGIN)
    if closest_dist > distance_threshold:
        if len(waypoints) == 1:
            if current_to_anchor_dist > BACKTRACK_MARGIN:
                add_waypoint(waypoints, current_pts, frame_idx=None, gray=frame_gray)
                logging.info("➕ Добавлена первая дочерняя точка (движение от старта)")
        else:
            last_center = waypoints[-1]['center']
            last_to_anchor = np.linalg.norm(last_center - anchor_center)
            if current_to_anchor_dist > last_to_anchor + BACKTRACK_MARGIN:
                add_waypoint(waypoints, current_pts, frame_idx=None, gray=frame_gray)
                logging.info("➕ Добавлена новая точка (удаление от старта)")

    # возврат: если ближайшая точка не стартовая и очень близко (гистерезис)
    elif closest_idx > 0 and closest_dist < (distance_threshold - HYSTERESIS_MARGIN):
        # обрезаем список до closest_idx (включительно)
        del waypoints[closest_idx+1:]
        logging.info(f"🔙 Возврат к точке {closest_idx}. Удалены последующие.")
        # пытаемся восстановить трекинг на текущем кадре: вернём список waypoints
    return waypoints

# ----------------- Расчёт угла через affine -----------------
def estimate_transform_and_angle(prev_pts, curr_pts):
    """
    Использует estimateAffinePartial2D с RANSAC для отказоустойчивой оценки
    Возвращает: dx, dy, angle(deg), inlier_count
    """
    if prev_pts is None or curr_pts is None:
        return 0.0, 0.0, 0.0, 0

    if len(prev_pts) < 3 or len(curr_pts) < 3:
        return 0.0, 0.0, 0.0, 0

    try:
        # ensure np.float32 Nx2
        p_prev = np.array(prev_pts, dtype=np.float32).reshape(-1,2)
        p_curr = np.array(curr_pts, dtype=np.float32).reshape(-1,2)
        # use RANSAC
        M, inliers = cv2.estimateAffinePartial2D(p_prev, p_curr, method=cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=2000)
        inlier_count = 0 if inliers is None else int(inliers.sum())

        if M is None:
            return 0.0, 0.0, 0.0, inlier_count

        # affine: [ a b tx; c d ty ]
        tx = M[0,2]
        ty = M[1,2]
        # rotation from matrix
        a = M[0,0]
        b = M[0,1]
        angle_rad = np.arctan2(b, a)  # note: acos(a) is less robust
        angle_deg = np.degrees(angle_rad)
        return float(tx), float(ty), float(angle_deg), inlier_count
    except Exception as e:
        logging.debug(f"[estimate_transform_and_angle] Failed: {e}")
        return 0.0, 0.0, 0.0, 0

# ----------------- main -----------------
def main():
    logging.info("🚀 Rope Ladder Tracker — improved")

    # Инициализация захвата в отдельном потоке
    grabber = FrameGrabber(0, IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX, TARGET_FPS)
    grabber.start()

    # Инициализируем детектор точек и Kalman
    feature_detector = FeatureDetector(MIN_FEATURES, MAX_FEATURES)
    kalman = SimpleKalman2D(process_noise=1e-3, measurement_noise=1e-1)

    # Параметры LK (подбирай под устройство)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        minEigThreshold=0.0001
    )

    waypoints = []
    anchor_center_fixed = None
    smoothed_curr_center = None
    prev_gray = None
    prev_tracked_pts = None
    tracked_pts = None

    tracking_active = False

    last_ladder_update_time = 0.0
    frame_idx = 0

    try:
        while True:
            ok, frame = grabber.read(timeout=1.0)
            if not ok or frame is None:
                time.sleep(FRAME_INTERVAL)
                continue

            frame_idx += 1

            # Применяем ROI, если задан
            if ROI is not None:
                x,y,w,h = ROI
                frame_roi = frame[y:y+h, x:x+w]
            else:
                frame_roi = frame

            # Перевод в серое и (опционально) UMat
            gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            # try UMat
            gray_for_processing = to_umat_if_available(gray)

            tracking_now = is_tracking_enabled()
            if not tracking_now:
                if tracking_active:
                    logging.info("🔴 Трекинг остановлен. Сбрасываем состояние.")
                    waypoints.clear()
                    anchor_center_fixed = None
                    tracking_active = False
                time.sleep(FRAME_INTERVAL)
                continue
            else:
                if not tracking_active:
                    # стартуем трекинг: детектируем начальные точки
                    pts0 = feature_detector.detect(gray)
                    if pts0 is None or len(pts0) < MIN_FEATURES:
                        logging.warning("Старт: недостаточно точек для инициализации. Пропуск.")
                        time.sleep(FRAME_INTERVAL)
                        continue
                    waypoints.clear()
                    # сохраняем исходную grayscale как эталон (numpy)
                    add_waypoint(waypoints, pts0, frame_idx=frame_idx, gray=gray.copy())
                    anchor_center_fixed = waypoints[0]['center'].copy()
                    prev_tracked_pts = pts0.copy()
                    tracked_pts = pts0.copy()
                    prev_gray = gray.copy()
                    smoothed_curr_center = np.mean(tracked_pts, axis=0)
                    kalman = SimpleKalman2D(process_noise=1e-3, measurement_noise=1e-1)
                    kalman.correct_and_predict(smoothed_curr_center)
                    tracking_active = True
                    logging.info(f"🟢 Трекинг запущен. Anchor at {anchor_center_fixed}")
                    continue  # переходим на следующий кадр

            # Если активен трекинг, пробуем LK относительно последнего wp.gray (или prev_gray)
            if tracking_active:
                # choose reference gray for optical flow (prefer last wp's gray if available)
                ref_gray = waypoints[-1]['gray'] if waypoints and waypoints[-1].get('gray') is not None else prev_gray
                if ref_gray is None:
                    ref_gray = prev_gray

                # Формируем точки для calcOpticalFlowPyrLK: Nx1x2 float32
                if tracked_pts is None or len(tracked_pts) == 0:
                    # попробуем найти новые точки на текущем кадре
                    new_pts = feature_detector.detect(gray)
                    if new_pts is None or len(new_pts) < MIN_FEATURES:
                        logging.warning("Потеря треков: нет новых точек.")
                        save_offset(0,0,0)
                        time.sleep(FRAME_INTERVAL)
                        continue
                    tracked_pts = new_pts.copy()

                try:
                    p0 = np.array(tracked_pts, dtype=np.float32).reshape(-1,1,2)
                    # calcOpticalFlowPyrLK принимает numpy images: если были UMat - использу numpy
                    p1, status, err = cv2.calcOpticalFlowPyrLK(ref_gray, gray, p0, None, **lk_params)
                    if p1 is None or status is None:
                        # попробуем обновить ключевые точки на текущем кадре
                        fresh = feature_detector.detect(gray)
                        if fresh is not None and len(fresh) >= MIN_FEATURES:
                            tracked_pts = fresh.copy()
                            prev_tracked_pts = tracked_pts.copy()
                            prev_gray = gray.copy()
                            logging.info("OpticalFlow вернул None -> использованы fresh точки.")
                            continue
                        else:
                            logging.warning("OpticalFlow провалился и fresh тоже нет.")
                            save_offset(0,0,0)
                            continue

                    good_idx = status.flatten() == 1
                    new_pts_valid = p1.reshape(-1,2)[good_idx]
                    prev_pts_valid = p0.reshape(-1,2)[good_idx]

                    if len(new_pts_valid) < 3 or len(prev_pts_valid) < 3:
                        # не хватает соответствий, найдём новые точки на кадре
                        fresh = feature_detector.detect(gray)
                        if fresh is not None and len(fresh) >= MIN_FEATURES:
                            tracked_pts = fresh.copy()
                            prev_tracked_pts = tracked_pts.copy()
                            prev_gray = gray.copy()
                            logging.info("Мало хороших индексов -> переинициализация точек.")
                            continue
                        else:
                            logging.warning("Мало соответствий и нет fresh -> сохраняем (0,0)")
                            save_offset(0,0,0)
                            continue

                    # оценим трансформацию между prev_pts_valid и new_pts_valid
                    tx, ty, angle_deg, inliers = estimate_transform_and_angle(prev_pts_valid, new_pts_valid)

                    # Центры
                    current_center = np.mean(new_pts_valid, axis=0)
                    if smoothed_curr_center is None:
                        smoothed_curr_center = current_center.copy()
                    else:
                        smoothed_curr_center = SMOOTHING_FACTOR * current_center + (1.0 - SMOOTHING_FACTOR) * smoothed_curr_center

                    # Kalman коррекция + предсказание
                    kalmed = kalman.correct_and_predict(smoothed_curr_center)
                    dx_px = kalmed[0] - waypoints[-1]['center'][0]
                    dy_px = kalmed[1] - waypoints[-1]['center'][1]

                    # save offsets
                    save_offset(dx_px, dy_px, angle=angle_deg)

                    # обновим tracked_pts и prev_gray для следующего шага
                    tracked_pts = new_pts_valid.copy()
                    prev_tracked_pts = tracked_pts.copy()
                    prev_gray = gray.copy()

                    # ladder management (раз в интервал)
                    now = time.time()
                    if now - last_ladder_update_time >= LADDER_UPDATE_INTERVAL:
                        rope_ladder_waypoint_management(waypoints, tracked_pts, distance_threshold=DISTANCE_THRESHOLD,
                                                        anchor_center_fixed=anchor_center_fixed, frame_gray=gray.copy())
                        last_ladder_update_time = now

                    # логирование
                    logging.debug(f"Inliers={inliers} | dx={dx_px:.2f} dy={dy_px:.2f} angle={angle_deg:.2f} | pts={len(tracked_pts)}")
                except Exception as e:
                    logging.warning(f"Ошибка во время расчёта OpticalFlow/Transform: {e}")
                    save_offset(0,0,0)
                    prev_gray = gray.copy()
                    time.sleep(FRAME_INTERVAL)
                    continue

            # небольшой sleep чтобы не перегружать цикл
            time.sleep(FRAME_INTERVAL * 0.2)

    except KeyboardInterrupt:
        logging.info("Остановлено пользователем")
    except Exception as e:
        logging.error(f"Ошибка: {e}", exc_info=True)
    finally:
        grabber.stop()
        logging.info("Система завершена")
    return 0

if __name__ == "__main__":
    exit(main())
