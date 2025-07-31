#!/usr/bin/env python3
"""
VRTX Rope Ladder Tracker
Оптимизировано для: RISC-V
"""

import cv2
import numpy as np
import time
import logging
import json
import os

# --- Настройки для слабых процессоров ---
IMAGE_WIDTH_PX = 640 
IMAGE_HEIGHT_PX = 480
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS

MIN_FEATURES = 50
MAX_FEATURES = 150
DISTANCE_THRESHOLD = 12.0   # порог добавления точки (пиксели)
BACKTRACK_MARGIN = 4.0      # запас при возврате

CLAHE_ENABLED = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

FLAG_PATH = '/home/orangepi/tracking_enabled.flag'  # путь можно поменять

# --- Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("rope_ladder.log"),
        logging.StreamHandler()
    ]
)

def is_tracking_enabled():
    """Проверяет флаг включения трекинга"""
    try:
        with open(FLAG_PATH, 'r') as f:
            return f.read().strip() == '1'
    except:
        return False

def save_offset(x_px, y_px, angle=0.0):
    """Сохраняет смещение в пикселях (как ожидает контроллер)"""
    data = {
        'x': int(x_px),
        'y': int(y_px),
        'angle': float(angle)
    }
    temp_file = 'offsets_tmp.json'
    final_file = 'offsets.json'
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_file, final_file)
    except Exception as e:
        logging.warning(f"❌ Не удалось сохранить offset: {e}")

def enhance_and_detect_features(gray):
    """Улучшение изображения и детекция точек (оптимизировано для слабого CPU)"""
    # Улучшение контраста
    if CLAHE_ENABLED:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        gray = clahe.apply(gray)

    # Адаптивный порог FAST в зависимости от контраста
    mean_val, std_val = cv2.meanStdDev(gray)
    base_threshold = 20
    # Повышаем порог при ярком свете, понижаем в тени/вечером
    threshold = max(10, min(40, int(base_threshold * (1.0 + (50 - std_val[0,0]) / 50))))

    # FAST детектор
    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(threshold)
    fast.setNonmaxSuppression(True)
    points = fast.detect(gray, None)

    if points is None or len(points) == 0:
        return None

    # Оставить только центральные точки (избежать краёв)
    pts = np.array([[p.pt[0], p.pt[1]] for p in points])
    h, w = gray.shape
    margin = 15
    mask = (pts[:, 0] > margin) & (pts[:, 0] < w - margin) & \
           (pts[:, 1] > margin) & (pts[:, 1] < h - margin)
    pts = pts[mask]

    # Ограничить количество точек
    if len(pts) > MAX_FEATURES:
        scores = np.array([cv2.FastFeatureDetector_create().compute(gray, [cv2.KeyPoint(x, y, 3) for x, y in pts])[1]])
        idx = np.argsort(scores[0])[::-1][:MAX_FEATURES]
        pts = pts[idx]

    return pts.reshape(-1, 1, 2).astype(np.float32) if len(pts) > 0 else None

def add_waypoint(waypoints, points, frame_idx=None):
    """Добавляет точку на лестницу"""
    if points is None or len(points) == 0:
        return
    center = np.mean(points.reshape(-1, 2), axis=0)
    wp = {
        'frame': frame_idx,
        'points': np.array(points, copy=True),
        'center': center
    }
    waypoints.append(wp)

def rope_ladder_waypoint_management(waypoints, current_points, distance_threshold=DISTANCE_THRESHOLD):
    """Управление точками по принципу верёвочной лестницы"""
    if len(waypoints) == 0 or current_points is None or len(current_points) == 0:
        return waypoints

    curr_center = np.mean(current_points.reshape(-1, 2), axis=0)
    anchor_center = waypoints[0]['center']

    # Поиск ближайшей точки
    closest_idx = 0
    closest_dist = np.linalg.norm(waypoints[0]['center'] - curr_center)
    for i, wp in enumerate(waypoints):
        dist = np.linalg.norm(wp['center'] - curr_center)
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = i

    # Удаление при возврате
    if closest_idx > 0 and closest_dist < distance_threshold:
        waypoints[:] = waypoints[:closest_idx + 1]
        logging.info(f"🔙 Возврат к точке {closest_idx}. Удалены последующие.")
        return waypoints

    # Добавление при удалении от старта
    if closest_dist > distance_threshold:
        last_to_anchor = 0.0
        if len(waypoints) > 1:
            last_center = waypoints[-1]['center']
            last_to_anchor = np.linalg.norm(last_center - anchor_center)
        current_to_anchor = np.linalg.norm(curr_center - anchor_center)
        if current_to_anchor > last_to_anchor + BACKTRACK_MARGIN:
            add_waypoint(waypoints, current_points)
            logging.info(f"➕ Добавлена новая точка (удаление от старта)")
    return waypoints

def main():
    logging.info("🪜 Rope Ladder Tracker: запуск для слабого CPU (Luckfox)")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH_PX)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT_PX)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # минимизировать задержку

    if not cap.isOpened():
        logging.error("❌ Не удалось открыть камеру.")
        return 1

    ret, frame = cap.read()
    if not ret or frame is None:
        logging.error("❌ Не удалось получить первый кадр.")
        return 1

    # Ресайз сразу, если камера даёт больше
    frame = cv2.resize(frame, (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracked_points = enhance_and_detect_features(gray)

    if tracked_points is None or len(tracked_points) < MIN_FEATURES:
        logging.error("❌ Недостаточно точек при старте.")
        return 1

    prev_gray = gray.copy()
    frame_idx = 0

    # 🪜 Верёвочная лестница
    waypoints = []
    add_waypoint(waypoints, tracked_points, frame_idx=frame_idx)

    # ⚙️ LK параметры (лёгкие)
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )

    fps = 0.0
    frame_count = 0
    start_time = time.time()

    tracking_active = False

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("⚠️ Пустой кадр — пропуск")
                time.sleep(FRAME_INTERVAL)
                continue

            frame_idx += 1
            frame = cv2.resize(frame, (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Проверка флага
            tracking_now = is_tracking_enabled()
            if not tracking_now:
                if tracking_active:
                    logging.info("🔴 Трекинг остановлен. Сброс waypoints.")
                    waypoints.clear()
                save_offset(0, 0)
                tracking_active = False
                prev_gray = gray
                time.sleep(FRAME_INTERVAL)
                continue
            else:
                if not tracking_active:
                    logging.info("🟢 Трекинг включён. Перезапуск с текущего кадра.")
                    fresh_points = enhance_and_detect_features(gray)
                    if fresh_points is not None and len(fresh_points) >= MIN_FEATURES:
                        waypoints.clear()
                        add_waypoint(waypoints, fresh_points, frame_idx=0)
                        tracked_points = fresh_points.copy()
                        logging.info("🔄 Новый старт установлен.")
                        tracking_active = True
                    else:
                        logging.warning("⚠️ Нет точек для старта — пропуск")
                        save_offset(0, 0)
                        prev_gray = gray
                        continue
                tracking_active = True

            # Отслеживание
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, tracked_points, None, **lk_params
            )
            if new_points is None or status is None:
                continue

            good_indices = [i for i, s in enumerate(status) if s == 1]
            tracked_points = new_points[good_indices]

            prev_gray = gray

            if len(tracked_points) == 0:
                save_offset(0, 0)
                logging.warning("⚠️ Нет точек — сброс")
                continue

            # 🪜 Управление лестницей
            rope_ladder_waypoint_management(waypoints, tracked_points)

            # 🏠 Проверка возврата
            try:
                current_center = np.mean(tracked_points.reshape(-1, 2), axis=0)
                anchor_center = waypoints[0]['center']
                dist_to_start = np.linalg.norm(current_center - anchor_center)
            except:
                save_offset(0, 0)
                continue

            if dist_to_start < DISTANCE_THRESHOLD:
                save_offset(0, 0)
                logging.info(f"🎯 ВОЗВРАТ В СТАРТ! (dist={dist_to_start:.1f}px)")
            else:
                dx_px = anchor_center[0] - current_center[0]
                dy_px = anchor_center[1] - current_center[1]
                save_offset(dx_px, dy_px)

            # FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                logging.info(f"📊 {fps:.1f} FPS | dx={int(dx_px):+6d} | dy={int(dy_px):+6d} | WPs={len(waypoints)}")
                frame_count = 0
                start_time = time.time()

            loop_time = time.time() - loop_start
            if loop_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - loop_time)

    except KeyboardInterrupt:
        logging.info("🛑 Остановлено пользователем.")
    except Exception as e:
        logging.error(f"💥 Ошибка: {e}", exc_info=True)
    finally:
        cap.release()
        logging.info("👋 Система завершена.")

    return 0

if __name__ == "__main__":
    exit(main())