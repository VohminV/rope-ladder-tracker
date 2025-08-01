#!/usr/bin/env python3
"""
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
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS

# --- Параметры трекинга и лестницы ---
MIN_FEATURES = 20
DISTANCE_THRESHOLD = 25.0
BACKTRACK_MARGIN = 15.0
HYSTERESIS_MARGIN = 10.0
LADDER_UPDATE_INTERVAL = 0.5
SMOOTHING_FACTOR = 0.7
FLAG_PATH = 'tracking_enabled.flag'

# --- Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rope_ladder.log", mode='w', encoding='utf-8')
    ]
)

# --- Функции ---
def save_offset(dx_m, dy_m, angle=0.0):
    """Сохраняет смещение в JSON файл"""
    x_px = int(dx_m * FOCAL_LENGTH_X)
    y_px = int(dy_m * FOCAL_LENGTH_Y)
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

def is_tracking_enabled():
    """Проверяет, активен ли трекинг (например, через внешний файл или сигнал)"""
    try:
        with open(FLAG_PATH, 'r') as f:
            return f.read().strip() == '1'
    except:
        return False

def adaptive_good_features(gray, min_features=20, max_features=1000):
    """Адаптивное обнаружение хороших точек с улучшенной стабильностью."""
    # 1. Улучшение контраста с CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    # 2. Дополнительная фильтрация для уменьшения шума
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 3. Расчет статистики для адаптивных параметров
    mean_val, std_val = cv2.meanStdDev(blurred)
    std_scalar = std_val[0,0]

    # 4. Адаптивный уровень качества
    quality_level = max(0.01, 0.1 * (1 - std_scalar / 50))

    height, width = gray.shape
    area = height * width

    # 5. Адаптивное количество точек
    num_features = max(min_features, min(max_features, int(area / 500)))

    # 6. Минимальное расстояние между точками
    min_distance = max(5, int(np.sqrt(area / num_features)))

    # 7. Использование детектора Харриса
    pts = cv2.goodFeaturesToTrack(
        image=blurred,
        maxCorners=num_features,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7,
        useHarrisDetector=True,
        k=0.04
    )

    # 8. Фолбэк на более простой детектор
    if pts is None or len(pts) < min_features:
        logging.debug(f"[adaptive_good_features] Harris failed, falling back. Found {len(pts) if pts is not None else 0} points.")
        pts = cv2.goodFeaturesToTrack(
            image=blurred,
            maxCorners=num_features,
            qualityLevel=0.01,
            minDistance=min_distance,
            blockSize=5,
            useHarrisDetector=False
        )

    # 9. Фолбэк на минимальное количество точек
    if pts is None or len(pts) < min_features:
        logging.warning(f"[adaptive_good_features] Not enough points even after fallback: {len(pts) if pts is not None else 0}")
        pts = cv2.goodFeaturesToTrack(
            image=blurred,
            maxCorners=min_features,
            qualityLevel=0.01,
            minDistance=3,
            blockSize=3
        )

    if pts is not None:
        return pts.reshape(-1).tolist()
    return None

def add_waypoint(waypoints, points, angle=None, frame_idx=None, gray=None):
    """Добавляет новую точку на лестнице и обновляет список точек для отслеживания"""
    logging.debug(f"[add_waypoint] Вызов: len(points)={len(points) if points is not None else 'None'}, MIN_FEATURES={MIN_FEATURES}")
    if points is None or len(points) < MIN_FEATURES * 2: # *2 потому что [x,y,x,y...]
        logging.debug(f"[add_waypoint] Не добавлена: недостаточно точек.")
        return
    
    # Обновляем список точек, которые будут использоваться для отслеживания
    global tracked_points
    if points is not None and len(points) >= MIN_FEATURES * 2:
        tracked_points = points.copy()
        logging.debug(f"[add_waypoint] Точки для отслеживания обновлены. Количество: {len(tracked_points)//2}")

    wp = {
        'frame': frame_idx,
        'points': np.array(points, copy=True),
        'angle': angle,
        'center': np.mean(np.array(points).reshape(-1, 2), axis=0),
        'gray': gray
    }
    waypoints.append(wp)
    logging.debug(f"[add_waypoint] Точка добавлена. Новый размер waypoints: {len(waypoints)}")

def rope_ladder_waypoint_management(waypoints, current_points, current_angle=None, distance_threshold=None, anchor_center_fixed=None, frame=None):
    """
    Управление точками по принципу верёвочной лестницы.
    """
    if len(waypoints) == 0:
        return waypoints

    try:
        curr_center = np.mean(np.array(current_points).reshape(-1, 2), axis=0)
    except (ValueError, TypeError) as e:
        logging.warning(f"[RLM] Ошибка вычисления curr_center: {e}")
        return waypoints

    # Используем anchor_center_fixed, если он передан
    anchor_center = anchor_center_fixed if anchor_center_fixed is not None else waypoints[0]['center']
    current_to_anchor_dist = np.linalg.norm(curr_center - anchor_center)

    # Поиск ближайшей существующей точки
    closest_dist = float('inf')
    closest_idx = -1
    for i, wp in enumerate(waypoints):
        dist = np.linalg.norm(wp['center'] - curr_center)
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = i

    # === ✅ Логика добавления новой точки ===
    if closest_dist > distance_threshold:
        if len(waypoints) == 1:
            # Первое удаление от стартовой точки
            last_to_anchor = 0.0
            if current_to_anchor_dist > BACKTRACK_MARGIN:
                add_waypoint(waypoints, current_points, current_angle, None, frame)
                logging.info(f"➕ Добавлена точка 1 (первое движение от старта)")
        else:
            # Уже есть хотя бы 2 точки
            last_center = waypoints[-1]['center']
            last_to_anchor = np.linalg.norm(last_center - anchor_center)
            if current_to_anchor_dist > last_to_anchor + BACKTRACK_MARGIN:
                add_waypoint(waypoints, current_points, current_angle, None, frame)
                logging.info(f"➕ Добавлена новая точка (удаление от старта)")

    # === 🔙 Логика возврата (с гистерезисом) ===
    elif closest_idx > 0 and closest_dist < (distance_threshold - HYSTERESIS_MARGIN):
        waypoints[:] = waypoints[:closest_idx + 1]
        # Ключевое изменение: Обновляем отслеживаемые точки НА ТЕКУЩЕМ кадре
        # А не из старых данных waypoints
        fresh_points = adaptive_good_features(frame) # <--- Используем текущий кадр
        if fresh_points is not None and len(fresh_points) >= MIN_FEATURES * 2:
            tracked_points = fresh_points.copy()
            logging.info(f"🔄 Трекинг восстановлен после возврата. Найдено {len(tracked_points)//2} точек.")
        else:
            logging.warning("⚠️ Не удалось найти новые точки после возврата.")
        logging.info(f"🔙 Возврат к точке {closest_idx}. Удалены последующие.")

    return waypoints

def calculate_angle(prev_points, curr_points):
    """
    Расчет угла поворота между двумя наборами точек.
    Использует все точки для повышения точности.
    Возвращает угол в радианах.
    """
    if len(prev_points) < 4 or len(curr_points) < 4:
        return 0.0

    try:
        prev_pts = np.array(prev_points).reshape(-1, 2)
        curr_pts = np.array(curr_points).reshape(-1, 2)
    except:
        return 0.0

    num_points = min(len(prev_pts), len(curr_pts))
    if num_points < 2:
        return 0.0

    prev_pts = prev_pts[:num_points]
    curr_pts = curr_pts[:num_points]

    prev_center = np.mean(prev_pts, axis=0)
    curr_center = np.mean(curr_pts, axis=0)

    prev_centered = prev_pts - prev_center
    curr_centered = curr_pts - curr_center

    H = np.dot(prev_centered.T, curr_centered)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = np.dot(Vt.T, U.T)

    cos_angle = R[0, 0]
    sin_angle = R[1, 0]
    angle_rad = np.arctan2(sin_angle, cos_angle)

    return angle_rad

def main():
    """Основная функция"""
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

    SHOW_DISPLAY = True
    if SHOW_DISPLAY:
        cv2.namedWindow("Rope Ladder Tracker", cv2.WINDOW_NORMAL)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracked_points = adaptive_good_features(gray)
    if tracked_points is None or len(tracked_points) < MIN_FEATURES * 2:
        logging.error("❌ Недостаточно точек при старте.")
        return 1

    prev_gray = gray.copy()
    prev_tracked_points = tracked_points.copy()
    frame_idx = 0

    waypoints = []
    anchor_center_fixed = None
    smoothed_curr_center = None

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    tracking_active = False
    dx_px = None
    dy_px = None

    fps = 0.0
    frame_count = 0
    start_time = time.time()
    last_ladder_update_time = 0.0

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

            tracking_now = is_tracking_enabled()
            if not tracking_now:
                if tracking_active:
                    logging.info("🔴 Трекинг остановлен. Сброс waypoints.")
                    waypoints.clear()
                    anchor_center_fixed = None
                    save_offset(0, 0)
                    tracking_active = False
                time.sleep(FRAME_INTERVAL)
                continue
            else:
                if not tracking_active:
                    logging.info("🟢 Трекинг включён. Устанавливаем новый старт.")
                    fresh_points = adaptive_good_features(gray)
                    if fresh_points is not None and len(fresh_points) >= MIN_FEATURES * 2:
                        waypoints.clear()
                        anchor_center_fixed = None
                        smoothed_curr_center = None

                        add_waypoint(waypoints, fresh_points, frame_idx=0, gray=gray)
                        if waypoints:
                             anchor_center_fixed = waypoints[0]['center'].copy()
                             logging.info(f"📍 Фиксированный старт установлен: ({anchor_center_fixed[0]:.2f}, {anchor_center_fixed[1]:.2f})")

                        tracked_points = fresh_points.copy()
                        prev_tracked_points = fresh_points.copy()
                        logging.info("🔄 Новый старт установлен.")
                        prev_gray = gray.copy()
                        last_ladder_update_time = time.time()
                    else:
                        logging.warning("⚠️ Нет точек для старта — пропуск кадра")
                        save_offset(0, 0)
                        time.sleep(FRAME_INTERVAL)
                        continue

                    tracking_active = True

            if tracking_active and waypoints[-1]['points'] is not None and len(waypoints[-1]['points']) > 0:
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(waypoints[-1]['gray'], gray, np.array(waypoints[-1]['points']).reshape(-1, 1, 2).astype(np.float32), None, **lk_params)
                
                if new_points is None or status is None:
                     save_offset(0, 0)
                     logging.warning("⚠️ Ошибка Optical Flow — сохраняем (0, 0)")
                     prev_gray = gray
                     continue

                good_indices = [i for i, s in enumerate(status.flatten()) if s == 1]
                if len(good_indices) == 0:
                     save_offset(0, 0)
                     logging.warning("⚠️ Все точки потеряны — сохраняем (0, 0)")
                     prev_gray = gray
                     continue

                new_tracked_points = new_points[good_indices].reshape(-1).tolist()
                prev_tracked_points = tracked_points
                tracked_points = new_tracked_points
                prev_gray = gray

                if tracked_points is None or len(tracked_points) == 0:
                    save_offset(0, 0)
                    logging.warning("⚠️ Нет точек после Optical Flow — сохраняем (0, 0)")
                    time.sleep(FRAME_INTERVAL)
                    continue

                # === 🔄 Сглаживание текущего центра ===
                current_center_raw = np.mean(np.array(tracked_points).reshape(-1, 2), axis=0)
                if smoothed_curr_center is None:
                    smoothed_curr_center = current_center_raw.copy()
                else:
                    smoothed_curr_center = SMOOTHING_FACTOR * current_center_raw + (1 - SMOOTHING_FACTOR) * smoothed_curr_center

                # === 🔄 Расчет угла поворота ===
                current_angle_rad = calculate_angle(prev_tracked_points, tracked_points)

                # === 🪜 Управление "лестницей" (с ограничением по времени) ===
                current_time = time.time()
                ladder_updated = False
                if current_time - last_ladder_update_time >= LADDER_UPDATE_INTERVAL:
                    if len(waypoints) > 0:
                        rope_ladder_waypoint_management(waypoints, tracked_points, current_angle=None,  distance_threshold=DISTANCE_THRESHOLD, anchor_center_fixed=anchor_center_fixed, frame=gray)
                        last_ladder_update_time = current_time
                        ladder_updated = True

                # --- Расчет и сохранение смещения ---
                # Используем СГЛАЖЕННУЮ позицию без коррекции на угол
                start_center = anchor_center_fixed if anchor_center_fixed is not None else waypoints[0]['center']
                dx_px = smoothed_curr_center[0] - start_center[0]
                dy_px = smoothed_curr_center[1] - start_center[1]
                # Передаем угол отдельно
                save_offset(dx_px / FOCAL_LENGTH_X, dy_px / FOCAL_LENGTH_Y, angle=np.degrees(current_angle_rad))

                # --- Отображение (если нужно) ---
                if SHOW_DISPLAY:
                    display_frame = frame.copy()

                    # --- 1. Рисуем ТЕКУЩИЕ отслеживаемые точки (зеленые) ---
                    if tracked_points:
                        for i in range(0, len(tracked_points), 2):
                            x, y = int(tracked_points[i]), int(tracked_points[i+1])
                            cv2.circle(display_frame, (x, y), 3, (0, 255, 0), -1) # Зеленый
                        cv2.putText(display_frame, f"Tracked: {len(tracked_points)//2}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # --- 2. Рисуем ТОЧКИ ИЗ СТАРТОВОЙ ПУТЕВОЙ ТОЧКИ (Синие) ---
                    if len(waypoints) > 0 and anchor_center_fixed is not None:
                        start_wp_points = waypoints[0]['points']
                        # Преобразуем в массив numpy
                        start_pts = start_wp_points.reshape(-1, 2)
                        # Отрисовываем все точки из стартовой WP
                        for pt in start_pts:
                            x, y = int(pt[0]), int(pt[1])
                            # Проверяем, в пределах ли кадра
                            if 0 <= x < IMAGE_WIDTH_PX and 0 <= y < IMAGE_HEIGHT_PX:
                                cv2.circle(display_frame, (x, y), 3, (255, 0, 0), -1) # Синий
                        cv2.putText(display_frame, f"Anchored: {len(start_pts)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # --- 3. Рисуем ПУТЕВЫЕ ТОЧКИ (Красные и синие) ---
                    for i, wp in enumerate(waypoints):
                        cx, cy = int(wp['center'][0]), int(wp['center'][1])
                        color = (255, 0, 0) if i == 0 else (0, 0, 255)
                        cv2.circle(display_frame, (cx, cy), 5, color, -1)
                        cv2.putText(display_frame, f'WP{i}', (cx+5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    # --- 4. Информация ---
                    if dx_px is not None and dy_px is not None:
                        cv2.putText(display_frame, f"dx: {dx_px:>+5.0f}px", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(display_frame, f"dy: {dy_px:>+5.0f}px", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(display_frame, f"WPs: {len(waypoints)}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.imshow("Rope Ladder Tracker", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # --- FPS ---
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    if len(waypoints) > 0:
                        dx_m = (dx_px / FOCAL_LENGTH_X) if dx_px is not None else 0
                        dy_m = (dy_px / FOCAL_LENGTH_Y) if dy_px is not None else 0
                        logging.info(f"📊 {fps:.1f} FPS | dx={dx_m*1000:>+5.0f} | dy={dy_m*1000:>+5.0f} | WPs={len(waypoints)}")
                    else:
                        logging.info(f"📊 {fps:.1f} FPS | dx=    0 | dy=    0 | WPs={len(waypoints)}")
                    frame_count = 0
                    start_time = time.time()

            else:
                save_offset(0, 0)
                if SHOW_DISPLAY:
                     cv2.imshow("Rope Ladder Tracker", frame)
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