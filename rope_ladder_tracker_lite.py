#!/usr/bin/env python3
import cv2
import numpy as np
import time
import logging
import json
import os

# --- Настройки для слабых процессоров (Luckfox) ---
IMAGE_WIDTH_PX = 640
IMAGE_HEIGHT_PX = 480
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS

MIN_FEATURES = 50
MAX_FEATURES = 150
DISTANCE_THRESHOLD = 12.0        # порог добавления/возврата (пиксели)
BACKTRACK_MARGIN = 6.0           # запас для продвижения вперёд
RETURN_HYSTERESIS = 2.0          # гистерезис для возврата (чтобы не дёргался)

CLAHE_ENABLED = True
CLAHE_CLIP = 3.0
CLAHE_TILE = (8, 8)

# Приблизительный масштаб: для камеры с FOV 70° и высоты 50–100 м
# 1 пиксел ≈ 0.02–0.05 м на земле. Можно калибровать.
# Мы не конвертируем в метры, но фильтруем выбросы по скорости.
MAX_PIXEL_VELOCITY = 30  # max допустимое смещение центра за кадр (пиксели)

FLAG_PATH = '/home/orangepi/tracking_enabled.flag'

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
    """Создаёт и настраивает FAST один раз (экономия CPU)"""
    fast = cv2.FastFeatureDetector_create()
    fast.setNonmaxSuppression(True)
    return fast

def adaptive_clahe(gray, clip_limit=3.0):
    """Адаптивный CLAHE с ограничением по контрасту"""
    if gray.mean() < 40:
        # Очень тёмно — уменьшаем clip, чтобы не усиливать шум
        clip = min(clip_limit, 1.5)
    elif gray.mean() > 200:
        # Очень светло — умеренный CLAHE
        clip = min(clip_limit, 2.0)
    else:
        clip = clip_limit

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(gray)

def normalize_illumination(gray):
    """Локальная нормализация яркости (устойчивость к теням)"""
    # Блюр для оценки фоновой освещённости
    background = cv2.GaussianBlur(gray, (127, 127), 15)
    # Нормализуем: gray - background + 127
    normalized = cv2.addWeighted(gray, 1.0, background, -1.0, 127)
    return np.clip(normalized, 0, 255).astype(np.uint8)

def enhance_and_detect_features(gray, fast_detector):
    """Улучшение и детекция с упором на стабильность и равномерность"""
    if CLAHE_ENABLED:
        gray = adaptive_clahe(gray)
    gray = normalize_illumination(gray)

    # Динамический порог FAST
    mean_val = cv2.mean(gray)[0]
    if mean_val < 30:
        threshold = 10
    elif mean_val < 80:
        threshold = 15
    elif mean_val < 160:
        threshold = 20
    else:
        threshold = 25
    fast_detector.setThreshold(threshold)

    points = fast_detector.detect(gray, None)
    if not points:
        return None

    pts = np.array([[p.pt[0], p.pt[1]] for p in points], dtype=np.float32)

    # Фильтр по краям (с запасом 20 пикселей)
    h, w = gray.shape
    margin = 20
    mask = (pts[:, 0] > margin) & (pts[:, 0] < w - margin) & \
           (pts[:, 1] > margin) & (pts[:, 1] < h - margin)
    pts = pts[mask]

    if len(pts) == 0:
        return None

    # Фильтрация по углу (избегаем линий)
    if len(pts) > 10:
        # Вычисляем среднее расстояние до соседей
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(pts))
        np.fill_diagonal(dists, np.inf)
        nearest_dists = np.min(dists, axis=1)
        mean_nearest = np.mean(nearest_dists)
        # Отсеиваем слишком упорядоченные (линии, сетки)
        if np.std(nearest_dists) < 0.3 * mean_nearest and len(pts) > 100:
            # Вероятно, регулярная структура — уменьшаем
            idx = np.random.choice(len(pts), size=min(80, len(pts)), replace=False)
            pts = pts[idx]

    # Ограничиваем количество
    if len(pts) > MAX_FEATURES:
        scores = [cv2.FastFeatureDetector_create().compute(gray, [cv2.KeyPoint(x, y, 3)])[1][0] for x, y in pts]
        idx = np.argsort(scores)[::-1][:MAX_FEATURES]
        pts = pts[idx]

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

def rope_ladder_waypoint_management(waypoints, current_points, distance_threshold=DISTANCE_THRESHOLD):
    """Управление точками по принципу верёвочной лестницы с гистерезисом"""
    if len(waypoints) == 0 or current_points is None or len(current_points) == 0:
        return waypoints

    curr_center = np.mean(current_points.reshape(-1, 2), axis=0)
    anchor_center = waypoints[0]['center']

    # Поиск ближайшей точки в лестнице
    closest_idx = 0
    min_dist = np.inf
    for i, wp in enumerate(waypoints):
        dist = np.linalg.norm(wp['center'] - curr_center)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    # Возврат, если ближе порога
    if min_dist < distance_threshold and closest_idx > 0:
        waypoints[:] = waypoints[:closest_idx + 1]
        logging.info(f"🔙 Возврат к точке {closest_idx} (dist={min_dist:.1f}px)")
        return waypoints

    # Проверка на продвижение вперёд (относительно старта)
    last_center = waypoints[-1]['center']
    dist_last_to_anchor = np.linalg.norm(last_center - anchor_center)
    dist_curr_to_anchor = np.linalg.norm(curr_center - anchor_center)

    # Только если продвинулись дальше (с запасом)
    if dist_curr_to_anchor > dist_last_to_anchor + BACKTRACK_MARGIN:
        add_waypoint(waypoints, current_points)
        logging.info(f"➕ Новая точка: dist_to_start={dist_curr_to_anchor:.1f}px")

    return waypoints

def main():
    logging.info("🪜 Rope Ladder Tracker: улучшенная версия для Luckfox (Cortex-A7)")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH_PX)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT_PX)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logging.error("❌ Не удалось открыть камеру.")
        return 1

    ret, frame = cap.read()
    if not ret or frame is None:
        logging.error("❌ Не удалось получить первый кадр.")
        return 1

    frame = cv2.resize(frame, (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Создаём FAST один раз
    fast_detector = create_fast_detector()

    tracked_points = enhance_and_detect_features(gray, fast_detector)
    if tracked_points is None or len(tracked_points) < MIN_FEATURES:
        logging.error("❌ Недостаточно точек при старте.")
        return 1

    prev_gray = gray.copy()
    frame_idx = 0

    # 🪜 Верёвочная лестница
    waypoints = []
    add_waypoint(waypoints, tracked_points, frame_idx=0)

    # ⚙️ LK параметры (лёгкие)
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
    )

    # EMA фильтр для сглаживания смещения (устранение дребезга)
    alpha = 0.3  # сглаживание (меньше = стабильнее, больше = быстрее реакция)
    smoothed_dx, smoothed_dy = 0.0, 0.0

    fps = 0.0
    frame_count = 0
    start_time = time.time()

    tracking_active = False
    last_valid_center = None

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("⚠️ Пустой кадр — пропуск")
                time.sleep(FRAME_INTERVAL)
                continue

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
                    logging.info("🟢 Трекинг включён. Перезапуск.")
                    fresh_points = enhance_and_detect_features(gray, fast_detector)
                    if fresh_points is not None and len(fresh_points) >= MIN_FEATURES:
                        waypoints.clear()
                        add_waypoint(waypoints, fresh_points, frame_idx=0)
                        tracked_points = fresh_points.copy()
                        last_valid_center = np.mean(fresh_points.reshape(-1, 2), axis=0)
                        smoothed_dx = smoothed_dy = 0.0
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
                prev_gray = gray
                continue

            good_indices = [i for i, s in enumerate(status) if s == 1]
            tracked_points = new_points[good_indices]

            prev_gray = gray

            if len(tracked_points) < MIN_FEATURES:
                save_offset(0, 0)
                logging.warning("⚠️ Мало точек — сброс")
                continue

            current_center = np.mean(tracked_points.reshape(-1, 2), axis=0)

            # Защита от резких прыжков (выбросы)
            if last_valid_center is not None:
                velocity = np.linalg.norm(current_center - last_valid_center)
                if velocity > MAX_PIXEL_VELOCITY:
                    logging.warning(f"⚠️ Слишком быстрое движение ({velocity:.1f}px) — пропуск кадра")
                    continue
            last_valid_center = current_center

            # 🪜 Управление лестницей
            rope_ladder_waypoint_management(waypoints, tracked_points)

            # Проверка возврата к старту
            try:
                anchor_center = waypoints[0]['center']
                dist_to_start = np.linalg.norm(current_center - anchor_center)
            except:
                save_offset(0, 0)
                continue

            if dist_to_start < DISTANCE_THRESHOLD - RETURN_HYSTERESIS:
                dx_px, dy_px = 0.0, 0.0
                logging.info(f"🎯 ВОЗВРАТ В СТАРТ! (dist={dist_to_start:.1f}px)")
            else:
                dx_px = anchor_center[0] - current_center[0]
                dy_px = anchor_center[1] - current_center[1]

            # EMA фильтр для сглаживания
            smoothed_dx = alpha * dx_px + (1 - alpha) * smoothed_dx
            smoothed_dy = alpha * dy_px + (1 - alpha) * smoothed_dy

            save_offset(smoothed_dx, smoothed_dy)

            # FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                logging.info(f"📊 {fps:.1f} FPS | dx={int(smoothed_dx):+6d} | dy={int(smoothed_dy):+6d} | WPs={len(waypoints)}")
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