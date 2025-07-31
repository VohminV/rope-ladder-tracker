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

# Приблизительный масштаб: для камеры с FOV 70° и высоты 50–100 м
# 1 пиксел ≈ 0.02–0.05 м на земле. Можно калибровать.
# Мы не конвертируем в метры, но фильтруем выбросы по скорости.
MAX_PIXEL_VELOCITY = 30  # max допустимое смещение центра за кадр (пиксели)


FLAG_PATH = 'tracking_enabled.flag'  # путь можно поменять

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

def create_fast_detector():
    """Создаёт и настраивает FAST один раз"""
    fast = cv2.FastFeatureDetector_create()
    fast.setNonmaxSuppression(True)
    return fast

def adaptive_clahe(gray, clip_limit=3.0):
    """Адаптивный CLAHE с учётом яркости"""
    mean_val = gray.mean()
    if mean_val < 40:
        clip = min(clip_limit, 1.5)
    elif mean_val > 200:
        clip = min(clip_limit, 2.0)
    else:
        clip = clip_limit

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(gray)

def normalize_illumination(gray):
    """Локальная нормализация яркости"""
    background = cv2.GaussianBlur(gray, (127, 127), 15)
    normalized = cv2.addWeighted(gray, 1.0, background, -1.0, 127)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def enhance_and_detect_features(gray, fast_detector):
    """Улучшение и детекция точек """
    if CLAHE_ENABLED:
        gray = adaptive_clahe(gray)
    gray = normalize_illumination(gray)

    # Динамический порог FAST
    mean_val = cv2.mean(gray)[0]
    threshold = 10 if mean_val < 30 else \
                15 if mean_val < 80 else \
                20 if mean_val < 160 else 25
    fast_detector.setThreshold(threshold)

    points = fast_detector.detect(gray, None)
    if not points:
        return None

    # Преобразуем точки в массив координат
    pts = np.array([[p.pt[0], p.pt[1]] for p in points], dtype=np.float32)

    # Фильтр по краям
    h, w = gray.shape
    margin = 20
    mask = (pts[:, 0] > margin) & (pts[:, 0] < w - margin) & \
           (pts[:, 1] > margin) & (pts[:, 1] < h - margin)
    pts = pts[mask]

    if len(pts) == 0:
        return None

    # --- ВАЖНО: Оставить только keypoints, прошедшие фильтр ---
    # Перестраиваем responses только для оставшихся точек
    kept_keypoints = [points[i] for i in range(len(points)) if mask[i]]
    responses = np.array([kp.response for kp in kept_keypoints])

    # Сортируем по качеству
    indices = np.argsort(responses)[::-1]  # от лучшего к худшему

    # Ограничиваем количество
    if len(pts) > MAX_FEATURES:
        indices = indices[:MAX_FEATURES]

    # Теперь индексы соответствуют текущему pts
    pts = pts[indices]

    return pts.reshape(-1, 1, 2).astype(np.float32)
    
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

def rope_ladder_waypoint_management(waypoints, current_points, 
                                   distance_threshold=DISTANCE_THRESHOLD,
                                   hysteresis=3.0):
    if len(waypoints) == 0 or current_points is None or len(current_points) < MIN_FEATURES:
        return waypoints

    current_center = np.mean(current_points.reshape(-1, 2), axis=0)
    start_center = waypoints[0]['center']

    min_dist = np.inf
    closest_idx = 0
    for i, wp in enumerate(waypoints):
        dist = np.linalg.norm(wp['center'] - current_center)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    # Возврат только если далеко от последней точки
    if min_dist < distance_threshold - hysteresis:
        if closest_idx < len(waypoints) - 1:  # не в последнюю
            waypoints[:] = waypoints[:closest_idx + 1]
            logging.info(f"BACK: returned to waypoint {closest_idx} (dist={min_dist:.1f}px)")
        return waypoints

    # Продвижение вперёд
    last_center = waypoints[-1]['center']
    dist_current_from_start = np.linalg.norm(current_center - start_center)
    dist_last_from_start = np.linalg.norm(last_center - start_center)

    if dist_current_from_start > dist_last_from_start + BACKTRACK_MARGIN:
        add_waypoint(waypoints, current_points)
        logging.info(f"ADD: new waypoint (dist_to_start={dist_current_from_start:.1f}px)")

    return waypoints

def main():
    logging.info("Rope Ladder Tracker: optimized for weak CPU (Cortex-A7)")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH_PX)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT_PX)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logging.error("Failed to open camera.")
        return 1

    ret, frame = cap.read()
    if not ret or frame is None:
        logging.error("Failed to get first frame.")
        return 1

    frame = cv2.resize(frame, (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fast_detector = create_fast_detector()

    tracked_points = enhance_and_detect_features(gray, fast_detector)
    if tracked_points is None or len(tracked_points) < MIN_FEATURES:
        logging.error("Not enough features at start.")
        return 1

    prev_gray = gray.copy()
    frame_idx = 0

    waypoints = []
    add_waypoint(waypoints, tracked_points, frame_idx=0)

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
    )

    # EMA фильтр для сглаживания смещения
    alpha_offset = 0.3
    smoothed_dx, smoothed_dy = 0.0, 0.0

    # EMA фильтр для стабилизации центра (устранение дрожания)
    alpha_center = 0.4
    last_valid_center = None

    fps = 0.0
    frame_count = 0
    start_time = time.time()

    tracking_active = False

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("Empty frame — skipping")
                time.sleep(FRAME_INTERVAL)
                continue

            frame = cv2.resize(frame, (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Проверка флага трекинга
            tracking_now = is_tracking_enabled()
            if not tracking_now:
                if tracking_active:
                    logging.info("Tracking stopped. Resetting waypoints.")
                    waypoints.clear()
                save_offset(0, 0)
                tracking_active = False
                prev_gray = gray
                time.sleep(FRAME_INTERVAL)
                continue
            else:
                if not tracking_active:
                    logging.info("Tracking enabled. Restarting...")
                    fresh_points = enhance_and_detect_features(gray, fast_detector)
                    if fresh_points is not None and len(fresh_points) >= MIN_FEATURES:
                        waypoints.clear()
                        add_waypoint(waypoints, fresh_points, frame_idx=0)
                        tracked_points = fresh_points.copy()
                        raw_center = np.mean(fresh_points.reshape(-1, 2), axis=0)
                        last_valid_center = raw_center.copy()
                        smoothed_dx, smoothed_dy = 0.0, 0.0
                        logging.info("New start set.")
                        tracking_active = True
                    else:
                        logging.warning("No features for start — skipping")
                        save_offset(0, 0)
                        prev_gray = gray
                        continue
                tracking_active = True

            # Отслеживание точек
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, tracked_points, None, **lk_params
            )
            if new_points is None or status is None:
                prev_gray = gray
                continue

            good_indices = [i for i, s in enumerate(status) if s == 1]
            tracked_points = new_points[good_indices]

            prev_gray = gray

            if len(tracked_points) < MIN_FEATURES:
                save_offset(0, 0)
                logging.warning("Too few points — reset")
                continue

            # Сглаживание центра для устранения дрожания
            raw_center = np.mean(tracked_points.reshape(-1, 2), axis=0)

            # Проверка на резкие скачки
            if last_valid_center is not None:
                velocity = np.linalg.norm(raw_center - last_valid_center)
                if velocity > MAX_PIXEL_VELOCITY:
                    logging.warning(f"Too fast motion ({velocity:.1f}px) — skip frame")
                    continue

            if last_valid_center is None:
                current_center = raw_center.copy()
            else:
                current_center = alpha_center * raw_center + (1 - alpha_center) * last_valid_center
            last_valid_center = current_center

            # Управление верёвочной лестницей с гистерезисом
            rope_ladder_waypoint_management(waypoints, tracked_points, hysteresis=3.0)

            # Проверка возврата в старт
            try:
                start_center = waypoints[0]['center']
                dist_to_start = np.linalg.norm(current_center - start_center)
            except:
                save_offset(0, 0)
                continue

            if dist_to_start < DISTANCE_THRESHOLD - 3.0:
                dx_px, dy_px = 0.0, 0.0
                logging.info(f"RETURN TO START! (dist={dist_to_start:.1f}px)")
            else:
                dx_px = start_center[0] - current_center[0]
                dy_px = start_center[1] - current_center[1]

            # Сглаживаем смещение
            smoothed_dx = alpha_offset * dx_px + (1 - alpha_offset) * smoothed_dx
            smoothed_dy = alpha_offset * dy_px + (1 - alpha_offset) * smoothed_dy

            save_offset(smoothed_dx, smoothed_dy)

            # Логирование FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                logging.info(f"FPS: {fps:.1f} | dx: {int(smoothed_dx):+6d} | dy: {int(smoothed_dy):+6d} | WPs: {len(waypoints)}")
                frame_count = 0
                start_time = time.time()

            # Контроль FPS
            loop_time = time.time() - loop_start
            if loop_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - loop_time)

    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
    finally:
        cap.release()
        logging.info("System terminated.")

    return 0
    
if __name__ == "__main__":
    exit(main())