"""
Scuba Dance Detector — версия на трекинге движения рук.

Логика без использования позы лица: чисто по траекториям обеих кистей.
- Для каждой руки считаем «stillness» (статичная) и «motion» (активная).
- «Scuba» = одна рука статична в верхней половине кадра (зажимает нос),
  вторая активно осциллирует (плывёт).

Дополнительно рисуются трейлы движения каждой руки.

Управление:
    Q или ESC — выход.
    R         — сброс трейлов.
"""

from __future__ import annotations

import collections
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).parent
GIF_PATH = ROOT / "scuba.gif"
HAND_MODEL = ROOT / "hand_landmarker.task"

TRAIL_LEN = 32                  # Длина трейла (кадров).
ANALYSIS_WINDOW = 14            # Окно анализа движения (кадров).
STILL_SPREAD_MAX = 0.04         # Максимум разброса для «статичной» руки.
MOVING_SPREAD_MIN = 0.10        # Минимум разброса для «активной» руки.
OSCILLATION_MIN = 3             # Сколько разворотов нужно для осцилляции.
STILL_HAND_TOP_FRACTION = 0.6   # Статичная рука должна быть в верхних 60% кадра.
TRIGGER_HOLD_FRAMES = 4
GIF_PLAY_SECONDS = 4.0

HAND_WRIST_IDX = 0
HAND_INDEX_TIP_IDX = 8


def load_gif_frames(path: Path) -> list[np.ndarray]:
    frames = iio.imread(path, index=None)
    out: list[np.ndarray] = []
    for frame in frames:
        if frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.append(frame)
    return out


def overlay_gif(canvas: np.ndarray, gif_frame: np.ndarray) -> np.ndarray:
    h, w = canvas.shape[:2]
    target_h = int(h * 0.55)
    scale = target_h / gif_frame.shape[0]
    target_w = int(gif_frame.shape[1] * scale)
    resized = cv2.resize(gif_frame, (target_w, target_h))
    x = (w - target_w) // 2
    y = (h - target_h) // 2
    out = canvas.copy()
    out[y:y + target_h, x:x + target_w] = resized
    cv2.rectangle(out, (x - 4, y - 4), (x + target_w + 4, y + target_h + 4),
                  (0, 255, 255), 3)
    cv2.putText(out, "SCUBA DANCE!", (x, y - 16),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_trail(canvas: np.ndarray, points: collections.deque, color: tuple[int, int, int]) -> None:
    h, w = canvas.shape[:2]
    pts = list(points)
    for i in range(1, len(pts)):
        a = (int(pts[i - 1][0] * w), int(pts[i - 1][1] * h))
        b = (int(pts[i][0] * w), int(pts[i][1] * h))
        thick = max(1, int(3 * (i / len(pts))))
        cv2.line(canvas, a, b, color, thick, cv2.LINE_AA)


def analyse_motion(hist: collections.deque) -> dict:
    """Считает spread, скорость и количество разворотов траектории."""
    if len(hist) < ANALYSIS_WINDOW:
        return {"valid": False, "spread": 0.0, "speed": 0.0, "oscillations": 0,
                "mean_y": 0.0}
    pts = list(hist)[-ANALYSIS_WINDOW:]
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    spread = float((xs.max() - xs.min()) + (ys.max() - ys.min()))

    deltas = np.diff(np.column_stack([xs, ys]), axis=0)
    speed = float(np.linalg.norm(deltas, axis=1).mean())

    dx = np.sign(np.diff(xs))
    dy = np.sign(np.diff(ys))
    osc_x = int(np.sum(np.abs(np.diff(dx)) > 0))
    osc_y = int(np.sum(np.abs(np.diff(dy)) > 0))
    oscillations = max(osc_x, osc_y)

    return {"valid": True, "spread": spread, "speed": speed,
            "oscillations": oscillations, "mean_y": float(ys.mean())}


def main() -> None:
    for required in (GIF_PATH, HAND_MODEL):
        if not required.exists():
            raise FileNotFoundError(f"Не найден файл: {required}")

    print("[init] Загружаю GIF…")
    gif_frames = load_gif_frames(GIF_PATH)
    print(f"[init] Кадров в GIF: {len(gif_frames)}")

    print("[init] Загружаю модель рук…")
    base = mp_python.BaseOptions(model_asset_path=str(HAND_MODEL))
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    hand_lm = vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру (индекс 0)")

    trails: dict[str, collections.deque] = {
        "Left": collections.deque(maxlen=TRAIL_LEN),
        "Right": collections.deque(maxlen=TRAIL_LEN),
    }
    trail_colors = {"Left": (255, 200, 0), "Right": (0, 200, 255)}
    trigger_streak = 0
    gif_until = 0.0
    gif_idx = 0
    gif_last_advance = time.time()
    fps_t0 = time.time()
    fps_count = 0
    fps_value = 0.0
    t_start = time.time()

    print("[run] Камера запущена. Зажми нос одной рукой и помаши другой!")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - t_start) * 1000)

        res = hand_lm.detect_for_video(mp_image, ts_ms)

        seen_labels: set[str] = set()
        if res.hand_landmarks and res.handedness:
            for hand_lms, handed in zip(res.hand_landmarks, res.handedness):
                label = handed[0].category_name  # 'Left'/'Right'
                seen_labels.add(label)
                wrist = hand_lms[HAND_WRIST_IDX]
                idx_tip = hand_lms[HAND_INDEX_TIP_IDX]
                # Берём кончик указательного как «точку руки» — он стабильнее
                # для трекинга кистевых жестов.
                point = ((wrist.x + idx_tip.x) / 2, (wrist.y + idx_tip.y) / 2)
                trails[label].append(point)

                px = int(point[0] * w)
                py = int(point[1] * h)
                cv2.circle(frame, (px, py), 9, trail_colors[label], 2)
                cv2.putText(frame, label, (px + 12, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, trail_colors[label], 1)

        # Если руку потеряли — очищаем её историю, чтобы не врать.
        for label in ("Left", "Right"):
            if label not in seen_labels:
                trails[label].clear()
            draw_trail(frame, trails[label], trail_colors[label])

        analysis = {label: analyse_motion(trails[label]) for label in trails}

        still_label: str | None = None
        moving_label: str | None = None
        for label, a in analysis.items():
            if not a["valid"]:
                continue
            if a["spread"] < STILL_SPREAD_MAX and a["mean_y"] < STILL_HAND_TOP_FRACTION:
                still_label = label
            if a["spread"] > MOVING_SPREAD_MIN and a["oscillations"] >= OSCILLATION_MIN:
                moving_label = label

        # Одна и та же рука не может быть и статичной, и активной.
        if still_label == moving_label:
            still_label = moving_label = None

        scuba = bool(still_label and moving_label and still_label != moving_label)
        trigger_streak = trigger_streak + 1 if scuba else 0
        if trigger_streak >= TRIGGER_HOLD_FRAMES:
            gif_until = max(gif_until, time.time() + GIF_PLAY_SECONDS)

        # HUD
        y_off = 25
        for label, a in analysis.items():
            color = trail_colors[label]
            text = (f"{label[0]}: spread={a['spread']:.2f} "
                    f"speed={a['speed']:.3f} osc={a['oscillations']}")
            cv2.putText(frame, text, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)
            y_off += 20

        status_color = (0, 255, 0) if scuba else (60, 60, 200)
        cv2.putText(frame, f"still: {still_label}  moving: {moving_label}",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"streak: {trigger_streak}", (10, y_off + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        now = time.time()
        if now < gif_until:
            if now - gif_last_advance > 0.06:
                gif_idx = (gif_idx + 1) % len(gif_frames)
                gif_last_advance = now
            frame = overlay_gif(frame, gif_frames[gif_idx])

        fps_count += 1
        if now - fps_t0 >= 1.0:
            fps_value = fps_count / (now - fps_t0)
            fps_count = 0
            fps_t0 = now
        cv2.putText(frame, f"FPS: {fps_value:.1f}", (w - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Scuba Dance Detector — Hand Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("r"):
            for t in trails.values():
                t.clear()
            trigger_streak = 0
            print("[run] Трейлы сброшены.")

    cap.release()
    cv2.destroyAllWindows()
    hand_lm.close()


if __name__ == "__main__":
    main()
