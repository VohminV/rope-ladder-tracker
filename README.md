# 🪜 rope-ladder-tracker

**Возврат квадрокоптера домой по оптическому потоку — без GPS, без MCU. Только камера и умная логика.**

Этот проект реализует систему **визуального возврата к стартовой точке** для квадрокоптера с использованием камеры, направленной вниз. 
Вместо классического интегрирования оптического потока, применяется оригинальный алгоритм **"верёвочной лестницы" (rope ladder)**, который позволяет надёжно возвращаться домой даже при дрейфе и шуме.

---

## 🚀 Зачем это нужно?

Традиционные системы:
- **GPS** — не работают внутри помещений.
- **Оптический поток + инерциальная навигация** — страдают от дрейфа.

Наше решение:
- ✅ Работает **без GPS и внешних систем**
- ✅ Устойчиво к **дрейфу и шуму**
- ✅ Обнаруживает **возврат в стартовую точку**
- ✅ Сохраняет **относительное смещение** для контроллера
- ✅ Реализовано на **чистом OpenCV и Python**

---

## 🔧 Как это работает?

### 🪜 Алгоритм "верёвочной лестницы"

1. **Стартовая точка** — фиксируется как первая.
2. При **удалении от старта** — добавляются новые "ступеньки" (контрольные кадры).
3. При **возврате** — система находит ближайшую ступень и обрезает историю после неё.
4. При **возврате к старту** — отправляется смещение `(0, 0)` → квадрокоптер останавливается.

> 💡 Это как спуск по верёвочной лестнице: вы можете вернуться на любую ступень.

### 📦 Вход и выход
- **Вход**: видео с камеры (например, 640x480, 30 FPS).
- **Выход**: файл `offsets.json` с полями:
```json
  {
    "x": 45,
    "y": -30,
    "angle": 0.0
  }
```
Где x, y — смещение в пикселях от текущей позиции до стартовой.

### 📦 Установка
```
git clone https://github.com/VohminV/rope-ladder-tracker.git
cd rope-ladder-tracker
pip install opencv-python numpy
```

### ▶️ Запуск 
```bash
python rope_ladder_tracker.py
``` 
 
### 📝 Управление 
  * Создайте файл tracking_enabled.flag с содержимым 1, чтобы включить запись в offsets.json.
  * Высоту можно задать в height.txt (в метрах), хотя в текущей версии она не используется (работа в пикселях).
     
### 🖼️ Пример работы 
```
2025-07-30 09:52:50,117 [INFO] 🎯 ВОЗВРАТ В СТАРТ! (dist=3.7px)
2025-07-30 09:52:50,469 [INFO] 📊 25.4 FPS | dx=+0.000 | dy=+0.000 | WPs=7
```
### 🤝 Как использовать с контроллером 

Другой скрипт (например, управляющий моторами) читает offsets.json и: 
  * Если x=0, y=0 → квадрокоптер дома.
  * Иначе → корректирует положение по x, y.
    
### 📸 Требования к камере 
  Направлена вниз.
  Хорошее освещение.
  Текстурированная поверхность (не однородная).
  Рекомендуется: 640x480, 30 FPS.
  
### 📚 Будущее развитие 
  * Поддержка поворота (через гистограммы или угол между точками).
  * Фильтрация смещений (EMA, Калман).
  * Визуализация траектории.
  * Интеграция с ROS2.
  * Работа с LiDAR вместо предполагаемой высоты.

### 🙏 Автор 
Разработано: VohminV

Лицензия: MIT 
     
     
