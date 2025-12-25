import cv2
from ultralytics import YOLO

CLASS_NAMES = [
    "Брусья параллельные",
    "Кольца гимнастические",
    "Перекладина для отжиманий",
    "Перекладина для подтягиваний",
    "Рукоход",
    "Рукоход змеевик",
    "Скамья для отжиманий",
    "Скамья для пресса",
    "Скамья наклонная",
    "Тренажер пресса",
    "Тумба",
    "Шведская стенка",
]


def run_equipment_inference_on_video(
    input_video_path: str,
    output_video_path: str,
    model: YOLO,
    conf_thres: float = 0.5,
):
    """
    Обрабатывает видео: рисует bounding boxes и подписи над объектами.
    """

    # Открываем видео
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {input_video_path}")

    # Настройки для записи
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Обработка видео...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция
        results = model(frame, conf=conf_thres, verbose=False)

        # Рисуем результаты
        annotated_frame = results[0].plot(
            font_size=10,
            line_width=2,  # толщина линии бокса
            labels=True,
            boxes=True,
        )

        # Запись кадра
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Видео сохранено: {output_video_path}")
