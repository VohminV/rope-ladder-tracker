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

MIN_FEATURES = 20 # Уменьшено для большей гибкости
DISTANCE_THRESHOLD = 25.0 # порог добавления новой точки (пиксели) - увеличено
BACKTRACK_MARGIN = 15.0   # минимальное "продвижение" назад - увеличено
HYSTERESIS_MARGIN = 10.0  # "мертвая зона" - увеличено

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

FLAG_PATH = 'tracking_enabled.flag'

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
        logging.warning(f"Не удалось сохранить offset: {e}")

def is_tracking_enabled():
    try:
        with open(FLAG_PATH, 'r') as f:
            return f.read().strip() == '1'
    except:
        return False

def adaptive_good_features(gray, min_features=20, max_features=1000):
    """Адаптивное обнаружение хороших точек"""
    # Apply CLAHE for contrast enhancement in low-light conditions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)

    # Calculate adaptive parameters based on image characteristics
    mean_val, std_val = cv2.meanStdDev(enhanced) # Используем enhanced вместо gray
    std_scalar = std_val[0,0]

    # Adaptive quality level based on image contrast
    quality_level = max(0.01, 0.1 * (1 - std_scalar / 50))

    height, width = gray.shape
    area = height * width

    # Adaptive feature count based on image area
    num_features = max(min_features, min(max_features, int(area / 500)))
    min_distance = max(5, int(np.sqrt(area / num_features)))

    # Detect features with enhanced image
    pts = cv2.goodFeaturesToTrack(
        image=enhanced, # Явно указываем параметр
        maxCorners=num_features,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7
    )
    if pts is not None:
        # Преобразуем в плоский список [x1, y1, x2, y2, ...]
        return pts.reshape(-1).tolist()
    return None

def add_waypoint(waypoints, points, angle=None, frame_idx=None):
    """Добавляет новую точку на лестнице"""
    logging.debug(f"[add_waypoint] Вызов: len(points)={len(points) if points is not None else 'None'}, MIN_FEATURES={MIN_FEATURES}")
    if points is None or len(points) < MIN_FEATURES * 2: # *2 потому что [x,y,x,y...]
        logging.debug(f"[add_waypoint] Не добавлена: недостаточно точек.")
        return
    wp = {
        'frame': frame_idx,
        'points': np.array(points, copy=True),
        'angle': angle,
        'center': np.mean(np.array(points).reshape(-1, 2), axis=0)
    }
    waypoints.append(wp)
    logging.debug(f"[add_waypoint] Точка добавлена. Новый размер waypoints: {len(waypoints)}")


def rope_ladder_waypoint_management(waypoints, current_points, current_angle=None, distance_threshold=DISTANCE_THRESHOLD, anchor_center_fixed=None):
    """
    Управление точками по принципу верёвочной лестницы.
    """
    # 1. Проверка на пустой список путевых точек
    if len(waypoints) == 0:
        logging.debug("[RLM] Список waypoints пуст, выход.")
        return waypoints

    # 2. Вычисление текущего центра и расстояния до стартовой точки (якоря)
    try:
        curr_center = np.mean(np.array(current_points).reshape(-1, 2), axis=0)
    except (ValueError, TypeError) as e:
        logging.warning(f"[RLM] Ошибка вычисления curr_center: {e}")
        return waypoints

    # Используем фиксированный якорь, если он передан и валиден
    anchor_center = None
    if anchor_center_fixed is not None and isinstance(anchor_center_fixed, np.ndarray) and anchor_center_fixed.shape == (2,):
        anchor_center = anchor_center_fixed
        logging.debug(f"[RLM] Используется ФИКСИРОВАННЫЙ anchor_center: ({anchor_center[0]:.2f}, {anchor_center[1]:.2f})")
    else:
        # fallback к waypoints[0], но логируем предупреждение
        anchor_center = waypoints[0]['center']
        logging.debug(f"[RLM] Используется waypoints[0] как anchor_center: ({anchor_center[0]:.2f}, {anchor_center[1]:.2f})")

    current_to_anchor_dist = np.linalg.norm(curr_center - anchor_center)

    # 3. Поиск ближайшей существующей точки
    closest_dist = float('inf')
    closest_idx = -1
    for i, wp in enumerate(waypoints):
        dist = np.linalg.norm(wp['center'] - curr_center)
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = i

    # === Подробное логирование текущего состояния ===
    logging.debug(
        f"[RLM] --- Состояние --- len(waypoints)={len(waypoints)}, "
        f"curr_center=({curr_center[0]:.2f}, {curr_center[1]:.2f}), "
        f"closest_dist={closest_dist:.2f}, closest_idx={closest_idx}, "
        f"current_to_anchor_dist={current_to_anchor_dist:.2f}, "
        f"distance_threshold={distance_threshold}"
    )

    # === ✅ Логика добавления новой точки ===
    # Условие 1: Достаточно далеко от ЛЮБОЙ существующей точки
    if closest_dist > distance_threshold:
        logging.debug(f"[RLM] Условие 1 выполнено: closest_dist ({closest_dist:.2f}) > distance_threshold ({distance_threshold})")

        if len(waypoints) == 1:
            # 4a. Первое движение от стартовой точки
            logging.debug(f"[RLM] len(waypoints) == 1")
            # last_to_anchor = 0.0 (расстояние от старта до старта)
            # Условие 2a: Достаточно далеко от стартовой точки (якоря)
            logging.debug(f"[RLM] Проверка условия 2a: current_to_anchor_dist ({current_to_anchor_dist:.2f}) > BACKTRACK_MARGIN ({BACKTRACK_MARGIN}) ?")
            if current_to_anchor_dist > BACKTRACK_MARGIN:
                logging.debug(f"[RLM] Условие 2a выполнено.")
                add_waypoint(waypoints, current_points, current_angle, None)
                logging.info(f"➕ Добавлена точка 1 (первое движение от старта)")
                logging.debug(f"[RLM] После добавления точки 1: len(waypoints)={len(waypoints)}")
            else:
                logging.debug(f"[RLM] Условие 2a НЕ выполнено.")

        else:
            # 4b. Последующие движения (уже есть минимум 2 точки)
            logging.debug(f"[RLM] len(waypoints) > 1")
            last_center = waypoints[-1]['center'] # Центр последней добавленной точки

            # Используем anchor_center (который теперь фиксирован или нет) для расчета last_to_anchor
            last_to_anchor = np.linalg.norm(last_center - anchor_center) # Расст. от последней точки до старта
            logging.debug(
                f"[RLM] last_center=({last_center[0]:.2f}, {last_center[1]:.2f}), "
                f"last_to_anchor (от якоря)={last_to_anchor:.2f}"
            )
            # Условие 2b: Текущая позиция значительно дальше от старта, чем последняя точка
            condition_2b = current_to_anchor_dist > last_to_anchor + BACKTRACK_MARGIN
            logging.debug(
                f"[RLM] Проверка условия 2b: current_to_anchor_dist ({current_to_anchor_dist:.2f}) "
                f"> (last_to_anchor + BACKTRACK_MARGIN) ({last_to_anchor + BACKTRACK_MARGIN:.2f}) ? "
                f"-> {condition_2b}"
            )
            if condition_2b:
                logging.debug(f"[RLM] Условие 2b выполнено.")
                add_waypoint(waypoints, current_points, current_angle, None)
                logging.info(f"Добавлена новая точка (удаление от старта)")
                logging.debug(f"[RLM] После добавления новой точки: len(waypoints)={len(waypoints)}")
            else:
                logging.debug(f"[RLM] Условие 2b НЕ выполнено.")

    # === 🔙 Логика возврата (с гистерезисом) ===
    # Условие 3: Есть к какой точке возвращаться (не к стартовой)
    # Условие 4: Достаточно близко к этой точке (с учетом гистерезиса)
    elif closest_idx > 0 and closest_dist < (distance_threshold - HYSTERESIS_MARGIN):
        logging.debug(f"[RLM] Условие возврата выполнено: closest_idx ({closest_idx}) > 0 И closest_dist ({closest_dist:.2f}) < (D-H) ({distance_threshold - HYSTERESIS_MARGIN})")
        # 5. Возврат: удаляем все точки после ближайшей
        waypoints[:] = waypoints[:closest_idx + 1] # Обрезаем список на месте
        logging.info(f"Возврат к точке {closest_idx}. Удалены последующие.")
        logging.debug(f"[RLM] После возврата: len(waypoints) теперь {len(waypoints)}")
    else:
        logging.debug(f"[RLM] Ничего не делаем.")

    return waypoints


def main():
    """Основная функция"""
    logging.info("Rope Ladder Tracker: возврат через структурированную историю")

    # --- Инициализация камеры ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH_PX)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT_PX)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        logging.error("Не удалось открыть камеру.")
        return 1

    ret, frame = cap.read()
    if not ret or frame is None:
        logging.error("Не удалось получить первый кадр.")
        return 1

    SHOW_DISPLAY = True
    if SHOW_DISPLAY:
        cv2.namedWindow("Rope Ladder Tracker", cv2.WINDOW_NORMAL)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracked_points = adaptive_good_features(gray)
    if tracked_points is None or len(tracked_points) < MIN_FEATURES * 2: # *2 для [x,y]
        logging.error("Недостаточно точек при старте.")
        return 1

    prev_gray = gray.copy()
    frame_idx = 0

    # === 🪜 Верёвочная лестница ===
    waypoints = []
    anchor_center_fixed = None # Новый: фиксированный якорь

    # === ⚙️ Параметры LK ===
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # === 🔁 Состояние трекинга ===
    tracking_active = False
    dx_px = None
    dy_px = None

    fps = 0.0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            loop_start = time.time()

            # - Захват кадра -
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("Пустой кадр — пропуск")
                time.sleep(FRAME_INTERVAL)
                continue

            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # - Проверка включения трекинга -
            tracking_now = is_tracking_enabled()
            if not tracking_now:
                if tracking_active:
                    logging.info("Трекинг остановлен. Сброс waypoints.")
                    waypoints.clear()
                    anchor_center_fixed = None # Сброс фиксированного якоря
                    save_offset(0, 0)
                    tracking_active = False
                # prev_gray = gray # 🔁 Обновляем для стабильности - не обновляем, если не активен
                time.sleep(FRAME_INTERVAL)
                continue
            else:
                if not tracking_active:
                    logging.info("Трекинг включён. Устанавливаем новый старт.")
                    # Перезапускаем с текущего кадра
                    fresh_points = adaptive_good_features(gray)
                    if fresh_points is not None and len(fresh_points) >= MIN_FEATURES * 2: # *2 для [x,y]
                        waypoints.clear()
                        anchor_center_fixed = None # Сброс перед новым стартом

                        add_waypoint(waypoints, fresh_points, frame_idx=0)
                        # Инициализируем фиксированный якорь после добавления первой точки
                        if waypoints:
                             anchor_center_fixed = waypoints[0]['center'].copy()
                             logging.info(f"Фиксированный старт установлен: ({anchor_center_fixed[0]:.2f}, {anchor_center_fixed[1]:.2f})")

                        tracked_points = fresh_points.copy() # ✅ Синхронизация
                        logging.info("Новый старт установлен.")
                        prev_gray = gray.copy() # Обновляем prev_gray только при новом старте
                    else:
                        logging.warning("Нет точек для старта — пропуск кадра")
                        save_offset(0, 0)
                        # prev_gray = gray # Не обновляем, если старт не удался
                        time.sleep(FRAME_INTERVAL)
                        continue

                    tracking_active = True
                    # Продолжаем выполнение ниже для обработки этого кадра

            # - Отслеживание -
            if tracking_active and tracked_points is not None and len(tracked_points) > 0:
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, np.array(tracked_points).reshape(-1, 1, 2).astype(np.float32), None, **lk_params)
                
                if new_points is None or status is None:
                     # Ошибка LK, сохраняем 0 и продолжаем
                     save_offset(0, 0)
                     logging.warning("Ошибка Optical Flow — сохраняем (0, 0)")
                     prev_gray = gray # Обновляем для следующего кадра
                     time.sleep(FRAME_INTERVAL)
                     continue

                good_indices = [i for i, s in enumerate(status.flatten()) if s == 1]
                if len(good_indices) == 0:
                     # Все точки потеряны, сохраняем 0 и продолжаем
                     save_offset(0, 0)
                     logging.warning("Все точки потеряны — сохраняем (0, 0)")
                     prev_gray = gray
                     time.sleep(FRAME_INTERVAL)
                     continue

                tracked_points = new_points[good_indices].reshape(-1).tolist()
                prev_gray = gray # ✅ Всегда обновляем

                # === 🔒 Защита от пустых точек ===
                if tracked_points is None or len(tracked_points) == 0:
                    save_offset(0, 0)
                    logging.warning("Нет точек после Optical Flow — сохраняем (0, 0)")
                    time.sleep(FRAME_INTERVAL)
                    continue

                # === 🪜 Управление "лестницей" ===
                if len(waypoints) > 0:
                    # Передаем фиксированный якорь
                    rope_ladder_waypoint_management(waypoints, tracked_points, current_angle=None, anchor_center_fixed=anchor_center_fixed)

                    # --- Расчет и сохранение смещения ---
                    if len(waypoints) > 0:
                        current_center = np.mean(np.array(tracked_points).reshape(-1, 2), axis=0)
                        start_center = anchor_center_fixed if anchor_center_fixed is not None else waypoints[0]['center']
                        dx_px = current_center[0] - start_center[0]
                        dy_px = current_center[1] - start_center[1]
                        save_offset(dx_px / FOCAL_LENGTH_X, dy_px / FOCAL_LENGTH_Y)
                    else:
                        dx_px, dy_px = 0, 0
                        save_offset(0, 0)

                    # --- Отображение (если нужно) ---
                    if SHOW_DISPLAY:
                        display_frame = frame.copy()
                        # Рисуем точки
                        for i in range(0, len(tracked_points), 2):
                            x, y = int(tracked_points[i]), int(tracked_points[i+1])
                            cv2.circle(display_frame, (x, y), 3, (0, 255, 0), -1)

                        # Рисуем waypoints
                        for i, wp in enumerate(waypoints):
                            cx, cy = int(wp['center'][0]), int(wp['center'][1])
                            color = (255, 0, 0) if i == 0 else (0, 0, 255)
                            cv2.circle(display_frame, (cx, cy), 5, color, -1)
                            cv2.putText(display_frame, f'WP{i}', (cx+5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                        # Инфо
                        if dx_px is not None and dy_px is not None:
                            cv2.putText(display_frame, f"dx: {dx_px:>+5.0f}px", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(display_frame, f"dy: {dy_px:>+5.0f}px", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(display_frame, f"WPs: {len(waypoints)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
                        logging.info(f"{fps:.1f} FPS | dx={dx_m*1000:>+5.0f} | dy={dy_m*1000:>+5.0f} | WPs={len(waypoints)}")
                    frame_count = 0
                    start_time = time.time()

            else:
                # Трекинг не активен или нет точек
                save_offset(0, 0)
                if SHOW_DISPLAY:
                     cv2.imshow("Rope Ladder Tracker", frame)
                     if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            # --- Интервал кадров ---
            loop_time = time.time() - loop_start
            if loop_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - loop_time)

    except KeyboardInterrupt:
        logging.info("Остановлено пользователем.")
    except Exception as e:
        logging.error(f"Ошибка: {e}", exc_info=True)
    finally:
        cap.release()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()
        logging.info("Система завершена.")
    return 0

if __name__ == "__main__":
    exit(main())