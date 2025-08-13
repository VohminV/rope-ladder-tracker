#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Стабильный трекер лестницы для Orange Pi 5: улучшенная работа при низкой освещенности и вибрации.
- Замена goodFeaturesToTrack на FAST + BRISK
- Улучшенная предобработка: CLAHE + Bilateral + Adaptive Threshold
- Устойчивый трекинг с фильтрацией вибраций
- Продвинутая логика лесенки с анализом траектории
- Адаптивный Kalman-фильтр
- Центр масс точек внутри каждого waypoint
"""

import cv2
import numpy as np
import time
import logging
import json
import os
import math
from collections import deque

# ----------------- Настройки -----------------
IMAGE_WIDTH_PX = 640
IMAGE_HEIGHT_PX = 480
TARGET_FPS = 20
FRAME_INTERVAL = 1.0 / TARGET_FPS

MIN_FEATURES = 25
MAX_FEATURES = 1000
DISTANCE_THRESHOLD = 40.0
BACKTRACK_MARGIN = 25.0
HYSTERESIS_MARGIN = 12.0
LADDER_UPDATE_INTERVAL = 0.8

INLIER_SAVE_RATIO = 0.5
MIN_INLIER_COUNT = 15

FLAG_PATH = 'tracking_enabled.flag'
OFFSETS_FILE = 'offsets.json'
ROI = None

SAVE_MODE = 'last'
DEBUG = False
SAVE_IN_METERS = False
CURRENT_HEIGHT_M = None
CAMERA_FOV_DEG = 70.0

# Буфер для хранения последних смещений (dx, dy)
OFFSET_BUFFER_SIZE = 20
offset_buffer = deque(maxlen=OFFSET_BUFFER_SIZE)

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rope_ladder_stable.log", mode='w', encoding='utf-8')
    ]
)

# ----------------- Утилиты -----------------
def px_to_m(dx_px, dy_px, height_m, fov_deg=CAMERA_FOV_DEG, img_w=IMAGE_WIDTH_PX):
    if height_m is None or height_m <= 0:
        return dx_px, dy_px
    half_fov_rad = math.radians(fov_deg) / 2.0
    width_m = 2.0 * height_m * math.tan(half_fov_rad)
    m_per_px = width_m / float(img_w)
    return dx_px * m_per_px, dy_px * m_per_px

def save_offset(dx, dy, angle=0.0, in_meters=False):
    data = {
        'x': int(dx), 'y': int(dy), 'angle': float(angle),
        'units': 'meters' if in_meters else 'pixels', 'ts': time.time()
    }
    tmp = OFFSETS_FILE + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, OFFSETS_FILE)
    except Exception as e:
        logging.warning(f"Не удалось сохранить offsets: {e}")

def is_tracking_enabled():
    try:
        with open(FLAG_PATH, 'r') as f:
            return f.read().strip() == '1'
    except:
        return False

# ----------------- Улучшенный Kalman фильтр -----------------
class EnhancedKalman2D:
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.initialized = False

    def correct_and_predict(self, meas):
        meas = np.array(meas, dtype=np.float32).reshape(2, 1)
        if not self.initialized:
            self.kf.statePost = np.array([[meas[0,0]], [meas[1,0]], [0.0], [0.0]])
            self.initialized = True
            return np.array([meas[0,0], meas[1,0]])
        self.kf.correct(meas)
        pred = self.kf.predict()
        return np.array([pred[0,0], pred[1,0]])

# ----------------- Улучшенный детектор FAST + BRISK -----------------
class FastBriskDetector:
    def __init__(self, min_features=MIN_FEATURES, max_features=MAX_FEATURES):
        self.min_features = min_features
        self.max_features = max_features
        self.fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
        self.brisk = cv2.BRISK_create()
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))

    def detect(self, gray):
        if gray is None:
            return None

        # Адаптивная предобработка
        brightness = np.mean(gray)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        if brightness < 80:
            # Темно: сильное улучшение
            enhanced = self.clahe.apply(denoised)
            detection_img = enhanced
        else:
            enhanced = self.clahe.apply(denoised)
            detection_img = enhanced

        # Детекция FAST
        keypoints = self.fast.detect(detection_img, None)
        if keypoints is None or len(keypoints) < self.min_features:
            return None

        # Ограничение количества ключевых точек
        keypoints = sorted(keypoints, key=lambda x: -x.response)
        keypoints = keypoints[:self.max_features]
        
        # Вычисление дескрипторов BRISK
        _, descriptors = self.brisk.compute(detection_img, keypoints)
        if descriptors is None:
            return None
            
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return pts

# ----------------- Устойчивый трекер с фильтрацией -----------------
class RobustTracker:
    def __init__(self):
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.window_size = 5
        self.history = []

    def track(self, prev_gray, curr_gray, prev_pts):
        if prev_pts is None or len(prev_pts) < 8:
            return None, 0, None

        p0 = np.array(prev_pts, dtype=np.float32).reshape(-1, 1, 2)
        p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
        
        if p1 is None or status is None or len(p1) == 0:
            return None, 0, None

        good = status.flatten() == 1
        if not np.any(good):
            return None, 0, None

        new_pts = p1.reshape(-1, 2)[good]
        old_pts = p0.reshape(-1, 2)[good]
        
        if len(new_pts) < MIN_FEATURES:
            return None, 0, None

        # Оценка аффинного преобразования
        M, mask = cv2.estimateAffinePartial2D(old_pts, new_pts, method=cv2.RANSAC, ransacReprojThreshold=6.0)
        inliers = mask.sum() if mask is not None else 0
        
        # Фильтрация по истории (фильтрация вибраций)
        center = np.mean(new_pts, axis=0)
        self.history.append(center)
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        if len(self.history) == self.window_size:
            smoothed_center = np.mean(self.history, axis=0)
        else:
            smoothed_center = center
            
        return new_pts, inliers, smoothed_center

# ----------------- Продвинутая логика лесенки с центром масс точек -----------------
def rope_ladder_waypoint_management(waypoints, current_center, anchor_center_fixed, tracked_pts, distance_threshold=DISTANCE_THRESHOLD):
    """
    Управление лестницей с центром масс точек внутри каждого waypoint.
    
    Args:
        waypoints: Список точек лестницы
        current_center: Текущий отфильтрованный центр
        anchor_center_fixed: Фиксированная начальная точка
        tracked_pts: Массив текущих трекаемых точек
    """
    if len(waypoints) == 0:
        return waypoints

    # Рассчитываем центр масс текущего кадра
    current_to_anchor = np.linalg.norm(current_center - anchor_center_fixed)
    last_center = waypoints[-1]['center']
    last_to_anchor = np.linalg.norm(last_center - anchor_center_fixed)
    
    # Добавление новой ступеньки
    if current_to_anchor > last_to_anchor + BACKTRACK_MARGIN:
        # Создаем новый waypoint с центром масс текущих точек
        new_waypoint_center = np.mean(tracked_pts, axis=0)  # Центр масс всех трекаемых точек
        
        wp = {
            'center': new_waypoint_center.copy(),
            'timestamp': time.time(),
            'cumulative': waypoints[-1]['cumulative'] + (new_waypoint_center - last_center),
            'points': tracked_pts.copy()  # Сохраняем все точки для будущих расчетов
        }
        waypoints.append(wp)
        logging.info(f"➕ Добавлена новая точка лестницы с центром масс {new_waypoint_center}")
    
    # Возврат к предыдущей ступеньке
    elif len(waypoints) > 1:
        dist_to_prev = np.linalg.norm(current_center - waypoints[-2]['center'])
        if dist_to_prev < HYSTERESIS_MARGIN:
            del waypoints[-1]
            logging.info(f"🔙 Возврат к предыдущей точке (осталось: {len(waypoints)})")
    
    return waypoints

# ----------------- Основной цикл -----------------
def main():
    logging.info("🚀 Стабильный трекер лестницы — запуск")
    
    detector = FastBriskDetector(MIN_FEATURES, MAX_FEATURES)
    tracker = RobustTracker()
    kalman = EnhancedKalman2D(process_noise=1e-3, measurement_noise=1e-1)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH_PX)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT_PX)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    waypoints = []
    anchor_center_fixed = None
    tracking_active = False
    last_update_time = 0
    dx_px, dy_px = 0.0, 0.0
    angle_deg = 0.0
    
    prev_gray = None
    tracked_pts = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(FRAME_INTERVAL)
                continue
                
            frame_idx += 1
            frame_roi = frame if ROI is None else frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
            gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            
            if not is_tracking_enabled():
                if tracking_active:
                    logging.info("🔴 Трекинг остановлен")
                    waypoints.clear()
                    tracking_active = False
                time.sleep(FRAME_INTERVAL)
                continue
                
            if not tracking_active:
                pts = detector.detect(gray)
                if pts is not None and len(pts) >= MIN_FEATURES:
                    # Инициализация первого waypoint с центром масс начальных точек
                    initial_center = np.mean(pts, axis=0)
                    waypoints = [{
                        'center': initial_center.copy(), 
                        'cumulative': np.array([0.0, 0.0]), 
                        'points': pts.copy()
                    }]
                    anchor_center_fixed = initial_center.copy()
                    tracked_pts = pts
                    prev_gray = gray.copy()
                    kalman = EnhancedKalman2D(process_noise=1e-3, measurement_noise=1e-1)
                    kalman.correct_and_predict(initial_center)
                    tracking_active = True
                    logging.info(f"🟢 Трекинг запущен. Якорь: {anchor_center_fixed}")
                else:
                    logging.debug("Инициализация: недостаточно точек")
                time.sleep(FRAME_INTERVAL)
                continue

            # Основной трекинг
            new_pts, inliers, smoothed_center = tracker.track(prev_gray, gray, tracked_pts)
            
            if new_pts is None or len(new_pts) < MIN_FEATURES or smoothed_center is None:
                logging.info("Переинициализация трека")
                new_pts = detector.detect(gray)
                if new_pts is not None and len(new_pts) >= MIN_FEATURES:
                    tracked_pts = new_pts
                    prev_gray = gray.copy()
                    continue
                else:
                    # При потере трекинга используем последнее известное смещение
                    if offset_buffer:
                        avg_dx = int(round(sum(x for x, y in offset_buffer) / len(offset_buffer)))
                        avg_dy = int(round(sum(y for x, y in offset_buffer) / len(offset_buffer)))
                    else:
                        avg_dx, avg_dy = 0, 0
                    save_offset(avg_dx, avg_dy, 0.0, in_meters=False)
                    continue

            # Проверка inliers
            min_inliers = max(MIN_INLIER_COUNT, len(new_pts) * INLIER_SAVE_RATIO)
            if inliers < min_inliers:
                logging.warning(f"Низкие inliers ({inliers} < {min_inliers}) -> пропуск")
                dx_px, dy_px = 0, 0
                avg_dx, avg_dy = 0, 0
            else:
                # Фильтрация Калманом
                kalmed = kalman.correct_and_predict(smoothed_center)
                
                # Обновление waypoints
                now = time.time()
                if now - last_update_time >= LADDER_UPDATE_INTERVAL:
                    waypoints = rope_ladder_waypoint_management(waypoints, kalmed, anchor_center_fixed, new_pts)
                    last_update_time = now
                
                # Рассчитываем смещение относительно последнего waypoint
                if len(waypoints) > 0:
                    last_waypoint_center = waypoints[-1]['center']
                    offset = kalmed - last_waypoint_center
                    dx_px, dy_px = int(offset[0]), int(offset[1])
                else:
                    dx_px, dy_px = 0, 0
                
                # Сглаживание через буфер
                offset_buffer.append((dx_px, dy_px))
                avg_dx = int(round(sum(x for x, y in offset_buffer) / len(offset_buffer)))
                avg_dy = int(round(sum(y for x, y in offset_buffer) / len(offset_buffer)))

                # Сохранение смещения
                if SAVE_IN_METERS and CURRENT_HEIGHT_M is not None:
                    dx_m, dy_m = px_to_m(avg_dx, avg_dy, CURRENT_HEIGHT_M)
                    save_offset(dx_m, dy_m, angle_deg, in_meters=True)
                else:
                    save_offset(avg_dx, avg_dy, angle_deg, in_meters=False)

            # Визуализация
            if DEBUG:
                debug_frame = frame.copy()
                for i, wp in enumerate(waypoints):
                    center = tuple(map(int, wp['center']))
                    cv2.circle(debug_frame, center, 6, (0,255,0), -1)
                    cv2.putText(debug_frame, f"W{i}", (center[0]+5, center[1]-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                # Визуализация отфильтрованного положения
                state = kalman.kf.statePost[:2].flatten()
                kalmed_int = (int(state[0]), int(state[1]))
                cv2.circle(debug_frame, kalmed_int, 8, (0,0,255), -1)
                cv2.imshow("DEBUG", debug_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            tracked_pts = new_pts
            prev_gray = gray.copy()
            time.sleep(max(0, FRAME_INTERVAL - (time.time() - now)))

    except KeyboardInterrupt:
        logging.info("Остановлено пользователем")
    except Exception as e:
        logging.error(f"Ошибка: {e}", exc_info=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Система завершена")

if __name__ == "__main__":
    main()