import cv2
import pickle
import os
from datetime import datetime

from person_tracker import process_video_for_lstm
from pose_preprocessor import PosePreprocessor
from inference import load_lstm_model, classify_person_sequence
from visualization import visualize_with_exercises


def full_process_with_loaded_models(video_path, lstm_model, yolo_model, device, min_visibility_sec=5, min_detection_percent=40):
    """
    Основная функция обработки видео
    """
    preprocessor = PosePreprocessor(seq_length=50, training=False)

    # 1. Трекинг - получаем только оригинальные координаты
    print(f"Трекинг видео: {video_path}")
    tracks = process_video_for_lstm(
        video_path=video_path,
        yolo_model=yolo_model,
        min_visibility_seconds=min_visibility_sec,
        min_detection_percent=min_detection_percent
    )
    
    clean_tracks = tracks
    if not tracks:
        print("Нет людей для анализа")
        return {}, {}

    # 2. Для каждого человека разбиваем на 2-секундные сегменты
    predictions = {}

    for track_id, track_data in tracks.items():
        print(f"\nОбработка человека ID {track_id}")
        # Используем ОРИГИНАЛЬНЫЕ координаты
        keypoints_original = track_data["keypoints_original"]
        timestamps = track_data["timestamps"]

        if len(timestamps) < 2:
            continue

        # Получаем FPS видео
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        cap.release()

        window_frames = int(2 * fps)  # 2 секунды
        step_frames = int(2 * fps)    # 2 секунда перекрытия

        segments = []

        for start in range(0, len(keypoints_original) - window_frames + 1, step_frames):
            end = start + window_frames

            segment = {
                "track_id": track_id,
                "start_time": float(timestamps[start]),
                "end_time": float(timestamps[end-1]),
                "keypoints": keypoints_original[start:end],  # Оригинальные координаты
                "segment_idx": len(segments)
            }
            segments.append(segment)

        print(f"  Сегментов: {len(segments)}")

        # 3. Классификация каждого сегмента
        segment_predictions = []
        
        class_names = [
            'barbell biceps curl', 'bench press', 'chest fly machine', 'deadlift', 
            'decline bench press', 'hammer curl', 'hip thrust', 'incline bench press', 
            'lat pulldown', 'lateral raise', 'leg extension', 'leg raises', 'other', 
            'plank', 'pull Up', 'push-up', 'romanian deadlift', 'russian twist', 
            'shoulder press', 'squat', 't bar row', 'tricep Pushdown', 'tricep dips'
        ]
        
        for segment in segments:
            prediction = classify_person_sequence(
                segment=segment,
                lstm_model=lstm_model,
                preprocessor=preprocessor,
                class_names=class_names,
                device=device
            )

            result = {
                "track_id": track_id,
                "segment_idx": segment["segment_idx"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "predicted_class": prediction["predicted_class"],
                "confidence": prediction["confidence"],
                "class_idx": prediction["class_idx"]
            }
            segment_predictions.append(result)

        predictions[track_id] = segment_predictions
    
    return clean_tracks, predictions


def save_results(tracks, predictions, output_dir="output"):
    """
    Сохраняет результаты в файлы
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_dir, timestamp)
    
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем tracks
    tracks_path = os.path.join(output_dir, "tracks.pkl")
    with open(tracks_path, 'wb') as f:
        pickle.dump(tracks, f)
    print(f"✅ Tracks сохранены: {tracks_path}")
    
    # Сохраняем predictions в pickle
    predictions_path = os.path.join(output_dir, "predictions.pkl")
    with open(predictions_path, 'wb') as f:
        pickle.dump(predictions, f)
    print(f"✅ Predictions сохранены: {predictions_path}")
    
    return output_dir, tracks_path, predictions_path


def process_exercise_video_with_loaded_models(
    video_path: str,
    lstm_model,
    yolo_model,
    device,
    output_dir: str = "output"
) -> tuple[str, str, str]:
    # Обработка видео
    tracks, predictions = full_process_with_loaded_models(
        video_path=video_path,
        lstm_model=lstm_model,
        yolo_model=yolo_model,
        device=device,
        min_visibility_sec=4,
        min_detection_percent=70
    )
    
    # Сохраняем результаты
    output_dir, tracks_path, predictions_path = save_results(tracks, predictions, output_dir)
    
    video_output = os.path.join(output_dir, "exercise_tracking_output.mp4")
    visualize_with_exercises(
        video_path=video_path,
        tracks=tracks,
        predictions=predictions,
        output_path=video_output
    )
    return output_dir, tracks_path, predictions_path, video_output

# Базовая обработка
# python main.py --video_path /path/to/video.mp4 --model_path /path/to/model.pth

# С визуализацией
# python main.py --video_path /path/to/video.mp4 --model_path /path/to/model.pth --visualize
