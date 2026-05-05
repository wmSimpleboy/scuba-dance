# Scuba Dance Detector

Камера + MediaPipe Hand Tracking — детектит «scuba dance» (одна рука зажимает нос, вторая активно двигается) и проигрывает GIF поверх видео.

## Demo

Жест: одна рука статично у носа, вторая руками плавательно осциллирует — на экран выезжает танцующий кот.

## Требования

- macOS / Linux / Windows
- Python 3.13+ (тестировано на 3.14)
- Веб-камера

## Установка

```bash
  git clone https://github.com/<you>/scuba-dance.git
  cd scuba-dance
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
```

Модели MediaPipe и GIF уже в репозитории.

## Запуск

```bash
source venv/bin/activate
python3 scuba_dance.py
```

## Управление

| Клавиша | Действие |
|---------|----------|
| `Q` / `ESC` | выход |
| `R` | сброс трейлов |

## Как работает

- `mediapipe.tasks.vision.HandLandmarker` находит обе кисти в реальном времени.
- Для каждой руки в скользящем окне 14 кадров считаются:
  - **spread** — суммарный разброс координат
  - **speed** — средняя скорость
  - **oscillations** — количество разворотов траектории
- «Scuba» = одна рука статична в верхней половине кадра + вторая активно осциллирует.
- При срабатывании ≥ 4 кадров подряд проигрывается GIF в течение 4 секунд.

## Тюнинг

Пороги детекции — константы в начале [scuba_dance.py](scuba_dance.py):

| Константа | Описание |
|-----------|----------|
| `STILL_SPREAD_MAX` | максимум разброса для «статичной» руки |
| `MOVING_SPREAD_MIN` | минимум разброса для «активной» |
| `OSCILLATION_MIN` | сколько разворотов нужно |
| `STILL_HAND_TOP_FRACTION` | в каких верхних % кадра должна быть статичная рука |
| `GIF_PLAY_SECONDS` | длительность проигрывания |
