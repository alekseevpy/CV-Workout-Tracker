import cv2
import numpy as np
from tqdm import tqdm

def visualize_with_exercises(video_path, tracks, predictions, output_path="tracking_with_exercises.mp4"):
    """
    Визуализация: скелеты разными цветами + подписи упражнений
    """
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return

    # Параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Создание визуализации: {width}x{height}, FPS: {fps:.1f}")

    # Создаем VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Цвета для разных треков (BGR формат в OpenCV)
    colors = [
        (0, 255, 0),    # зеленый
        (255, 0, 0),    # синий
        (0, 0, 255),    # красный
        (255, 255, 0),  # голубой
        (255, 0, 255),  # розовый
        (0, 255, 255),  # желтый
        (255, 255, 255),# белый
    ]

    # Создаем карту кадр→треки
    frame_map = {}
    for track_id, track_data in tracks.items():
        frame_indices = track_data.get("frame_indices", [])
        keypoints = track_data.get("keypoints_original", [])

        for i, frame_idx in enumerate(frame_indices):
            if i < len(keypoints):
                if frame_idx not in frame_map:
                    frame_map[frame_idx] = []
                frame_map[frame_idx].append((track_id, keypoints[i]))

    # Обрабатываем видео
    print("\nОбработка видео...")

    for frame_idx in tqdm(range(total_frames), desc="Кадры"):
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps if fps > 0 else 0

        # Рисуем треки в текущем кадре
        if frame_idx in frame_map:
            for track_id, keypoints in frame_map[frame_idx]:
                # Цвет для трека
                color = colors[track_id % len(colors)]

                # Рисуем скелет (упрощенный)
                if len(keypoints) == 51:
                    kp = keypoints.reshape(17, 3)
                else:
                    kp = keypoints

                # Соединения
                connections = [
                    (0, 1), (0, 2), (1, 3), (2, 4),
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (5, 11), (6, 12)
                ]

                # Рисуем линии
                for start_idx, end_idx in connections:
                    if start_idx < len(kp) and end_idx < len(kp):
                        x1, y1, c1 = kp[start_idx]
                        x2, y2, c2 = kp[end_idx]

                        if c1 > 0.3 and c2 > 0.3:
                            if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
                                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Рисуем точки
                for i, (x, y, conf) in enumerate(kp):
                    if conf > 0.3 and 0 <= x < width and 0 <= y < height:
                        cv2.circle(frame, (int(x), int(y)), 4, color, -1)

                # Подпись с упражнением
                exercise_text = f"ID:{track_id}"
                if predictions and track_id in predictions:
                    for segment in predictions[track_id]:
                        if segment["start_time"] <= current_time <= segment["end_time"]:
                            exercise_text = f"ID:{track_id} - {segment['predicted_class']}"
                            break

                # Позиция для текста
                visible = kp[kp[:, 2] > 0.3]
                if len(visible) > 0:
                    text_x = int(np.mean(visible[:, 0]))
                    text_y = int(np.min(visible[:, 1])) - 20

                    text_x = max(10, min(text_x, width - 200))
                    text_y = max(30, text_y)

                    # Фон
                    (w, h), _ = cv2.getTextSize(exercise_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame,
                                (text_x - 5, text_y - h - 5),
                                (text_x + w + 5, text_y + 5),
                                (0, 0, 0), -1)

                    # Текст
                    cv2.putText(frame, exercise_text,
                              (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Время и номер кадра
        time_str = f"{int(current_time//60):02d}:{current_time%60:05.2f}"
        cv2.putText(frame, f"Time: {time_str} | Frame: {frame_idx}/{total_frames}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Записываем кадр
        out.write(frame)

    # Освобождаем ресурсы
    cap.release()
    out.release()

    print(f"\n Визуализация сохранена: {output_path}")

    return output_path