import cv2
import numpy as np
import yaml
import tempfile
from ultralytics import YOLO
from collections import defaultdict

def process_video_for_lstm(video_path, yolo_model, min_visibility_seconds=5, min_detection_percent=40):
    """
    Трекинг людей на видео с фильтрацией

    Возвращает: {track_id: {"keypoints_original": [...], ...}}
    """
    cap = cv2.VideoCapture(video_path)

    tracker_config = {
        'tracker_type': 'bytetrack',
        'track_high_thresh': 0.6,
        'track_low_thresh': 0.3,
        'new_track_thresh': 0.7,
        'track_buffer': 10,
        'match_thresh': 0.70,
        'fuse_score': True
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(tracker_config, f)
        tracker_file = f.name

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    # Хранилище данных
    tracks = defaultdict(lambda: {
        "keypoints_original": [],  # ТОЛЬКО оригинальные координаты
        "timestamps": [],
        "frame_indices": [],
        "confidences": [],
        "bboxes": []
    })

    # Основной цикл трекинга
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция с трекингом
        results = yolo_model.track(frame, persist=True, verbose=False, conf=0.77, iou=0.5, tracker=tracker_file)

        # Обработка детекций
        if (results[0].boxes is not None and
            results[0].boxes.id is not None and
            results[0].keypoints is not None):

            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            keypoints_list = results[0].keypoints.data.cpu().numpy()

            for i, track_id in enumerate(track_ids):
                kps = keypoints_list[i]
                if len(kps.shape) == 3:
                    kps = kps[0]  # [17, 3]

                # Сохраняем ТОЛЬКО оригинальные координаты
                tracks[track_id]["keypoints_original"].append(kps.flatten())
                tracks[track_id]["timestamps"].append(frame_idx / fps)
                tracks[track_id]["frame_indices"].append(frame_idx)

        frame_idx += 1

    cap.release()

    # ФИЛЬТРАЦИЯ
    filtered_tracks = {}

    for track_id, data in tracks.items():
        if len(data["timestamps"]) == 0:
            continue

        # Вычисляем метрики
        start_time = data["timestamps"][0]
        end_time = data["timestamps"][-1]
        duration = end_time - start_time

        # Вычисляем процент кадров с детекцией
        detection_windows = []
        window_size = 1.0  # 1 секунда

        current_window_start = start_time
        while current_window_start < end_time:
            window_end = current_window_start + window_size

            # Кадры в этом окне
            frames_in_window = [
                i for i, t in enumerate(data["timestamps"])
                if current_window_start <= t < window_end
            ]

            if len(frames_in_window) > 0:
                detection_windows.append(len(frames_in_window))

            current_window_start = window_end

        # Если были окна с детекцией
        if detection_windows:
            avg_detection_in_window = np.mean(detection_windows)
            expected_frames_per_window = window_size * fps
            detection_percentage = (avg_detection_in_window / expected_frames_per_window) * 100
        else:
            detection_percentage = 0

        # Критерии фильтрации
        condition1 = (duration >= min_visibility_seconds and
                     detection_percentage >= min_detection_percent)
        condition2 = (detection_percentage >= 80)

        if condition1 or condition2:
            # Конвертируем в numpy
            filtered_tracks[track_id] = {
                "keypoints_original": np.array(data["keypoints_original"]),  # Только оригинальные
                "timestamps": np.array(data["timestamps"]),
                "frame_indices": np.array(data["frame_indices"]),
                "duration": duration,
                "detection_percentage": detection_percentage
            }
    
    return filtered_tracks