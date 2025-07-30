#!/usr/bin/env python3
"""
motion_drone_rope_ladder.py

Система визуального возврата по принципу "верёвочной лестницы":
- waypoints[0] = стартовая точка (anchor)
- Добавляем точку при движении вперёд (удалении от anchor)
- Удаляем точки при возврате (если совпадаем с существующей)
- Сохраняем (0,0) при возврате к waypoints[0]
"""

import cv2
import numpy as np
import time
import logging
import json
import os

# --- Настройки ---
FOCAL_LENGTH_X = 300
FOCAL_LENGTH_Y = 300
IMAGE_WIDTH_PX = 640
IMAGE_HEIGHT_PX = 480

MIN_FEATURES = 50
DISTANCE_THRESHOLD = 15.0       # порог добавления новой точки (пиксели)
BACKTRACK_MARGIN = 5.0           # запас при возврате
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS

FLAG_PATH = '/home/orangepi/tracking_enabled.flag'

# --- Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        #logging.FileHandler("/home/orangepi/motion_tracker.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def is_tracking_enabled():
    try:
        with open(FLAG_PATH, 'r') as f:
            return f.read().strip() == '1'
    except:
        return False

def save_offset(x_px, y_px, angle=0.0):
    """
    Сохраняет смещение в пикселях — как ожидает контроллер.
    """
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

def adaptive_good_features(gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    mean_val, std_val = cv2.meanStdDev(gray)
    std_scalar = std_val[0, 0]
    quality_level = max(0.01, 0.1 * (1 - std_scalar / 50))
    h, w = gray.shape
    area = h * w
    num_features = max(50, min(1000, area // 200))
    min_distance = max(5, int(np.sqrt(area / num_features)))
    points = cv2.goodFeaturesToTrack(
        enhanced,
        maxCorners=num_features,
        qualityLevel=float(quality_level),
        minDistance=min_distance,
        blockSize=7
    )
    return points

def add_waypoint(waypoints, points, angle=None, frame_idx=None):
    """Добавляет новую точку на лестнице"""
    if points is None or len(points) < MIN_FEATURES:
        return
    wp = {
        'frame': frame_idx,
        'points': np.array(points, copy=True),
        'angle': angle,
        'center': np.mean(np.array(points).reshape(-1, 2), axis=0)
    }
    waypoints.append(wp)

def rope_ladder_waypoint_management(waypoints, current_points, current_angle=None, distance_threshold=DISTANCE_THRESHOLD):
    """
    Управление точками по принципу верёвочной лестницы
    """
    if len(waypoints) == 0:
        return waypoints

    curr_center = np.mean(np.array(current_points).reshape(-1, 2), axis=0)
    anchor_center = waypoints[0]['center']
    current_to_anchor_dist = np.linalg.norm(curr_center - anchor_center)

    # Поиск ближайшей точки
    closest_dist = float('inf')
    closest_idx = -1
    for i, wp in enumerate(waypoints):
        dist = np.linalg.norm(wp['center'] - curr_center)
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = i

    # === ✅ Разрешаем добавление при len >= 1 ===
    if closest_dist > distance_threshold:
        if len(waypoints) == 1:
            # Первое удаление от стартовой точки
            last_to_anchor = 0.0
            if current_to_anchor_dist > BACKTRACK_MARGIN:
                add_waypoint(waypoints, current_points, current_angle, None)
                logging.info(f"➕ Добавлена точка 1 (первое движение от старта)")
        else:
            # Уже есть хотя бы 2 точки
            last_center = waypoints[-1]['center']
            last_to_anchor = np.linalg.norm(last_center - anchor_center)
            if current_to_anchor_dist > last_to_anchor + BACKTRACK_MARGIN:
                add_waypoint(waypoints, current_points, current_angle, None)
                logging.info(f"➕ Добавлена новая точка (удаление от старта)")
    # Возврат — удаляем хвост
    elif closest_idx > 0:
        waypoints[:] = waypoints[:closest_idx + 1]
        logging.info(f"🔙 Возврат к точке {closest_idx}. Удалены последующие.")

    return waypoints

def main():
    logging.info("🚀 Rope Ladder Tracker: возврат через структурированную историю")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH_PX)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT_PX)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        logging.error("❌ Не удалось открыть камеру.")
        return 1

    ret, frame = cap.read()
    if not ret or frame is None:
        logging.error("❌ Не удалось получить первый кадр.")
        return 1

    SHOW_DISPLAY = False
    if SHOW_DISPLAY:
        cv2.namedWindow("Rope Ladder Tracker", cv2.WINDOW_NORMAL)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracked_points = adaptive_good_features(gray)
    if tracked_points is None or len(tracked_points) < MIN_FEATURES:
        logging.error("❌ Недостаточно точек при старте.")
        return 1

    prev_gray = gray.copy()
    frame_idx = 0

    # === 🪜 Верёвочная лестница ===
    waypoints = []
    anchor_center = np.mean(np.array(tracked_points).reshape(-1, 2), axis=0)
    add_waypoint(waypoints, tracked_points, frame_idx=frame_idx)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    fps = 0.0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("⚠️ Пустой кадр — пропуск")
                time.sleep(FRAME_INTERVAL)
                continue

            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- Отслеживание ---
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, tracked_points, None, **lk_params)
            if new_points is None or status is None:
                tracked_points = adaptive_good_features(gray)
                if tracked_points is None:
                    tracked_points = np.array([])
                prev_gray = gray
                continue

            good_indices = [i for i, s in enumerate(status) if s == 1]
            if len(good_indices) < MIN_FEATURES:
                tracked_points = adaptive_good_features(gray)
                if tracked_points is None:
                    tracked_points = np.array([])
            else:
                tracked_points = new_points[good_indices]

            prev_gray = gray.copy()
            
            # === 🔒 Защита от пустых точек ===
            if tracked_points is None or len(tracked_points) == 0:
                # Сохраняем 0,0 — как "потеряли трекинг"
                save_offset(0, 0)
                logging.warning("⚠️ Нет точек — сохраняем (0, 0)")
                continue


            # === 🪜 Управление "лестницей" ===
            rope_ladder_waypoint_management(waypoints, tracked_points, current_angle=None)


            # === 🏠 Проверка: вернулись ли в старт? ===
            current_center = np.mean(np.array(tracked_points).reshape(-1, 2), axis=0)
            anchor_center = waypoints[0]['center']
            dist_to_start = np.linalg.norm(current_center - anchor_center)

            if dist_to_start < DISTANCE_THRESHOLD:
                save_offset(0, 0)
                logging.info(f"🎯 ВОЗВРАТ В СТАРТ! (dist={dist_to_start:.1f}px)")
            else:
                # Смещение от текущего к стартовому
                dx_px = anchor_center[0] - current_center[0]
                dy_px = anchor_center[1] - current_center[1]
                save_offset(int(dx_px), int(dy_px))

            # === 📊 FPS ===
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                logging.info(f"📊 {fps:.1f} FPS | dx={dx_px:+.3f}м | dy={dy_px:+.3f}м | WPs={len(waypoints)}")
                frame_count = 0
                start_time = time.time()

            # === 🖼️ Визуализация ===
            if SHOW_DISPLAY:
                display = frame.copy()
                status = "HOME" if dist_to_start < DISTANCE_THRESHOLD else f"WP {len(waypoints)-1}"
                color = (0, 255, 0) if dist_to_start < DISTANCE_THRESHOLD else (255, 165, 0)
                cv2.putText(display, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(display, f"WPs: {len(waypoints)}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                for pt in tracked_points:
                    cv2.circle(display, (int(pt[0][0]), int(pt[0][1])), 2, (255, 0, 0), -1)
                cv2.imshow("Rope Ladder Tracker", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            loop_time = time.time() - loop_start
            if loop_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - loop_time)

    except KeyboardInterrupt:
        logging.info("🛑 Остановлено пользователем.")
    except Exception as e:
        logging.error(f"💥 Ошибка: {e}", exc_info=True)
    finally:
        cap.release()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()
        logging.info("👋 Система завершена.")

    return 0

if __name__ == "__main__":
    exit(main())