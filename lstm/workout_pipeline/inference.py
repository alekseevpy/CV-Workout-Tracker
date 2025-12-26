import torch
import numpy as np
from pose_lstm import EnhancedPoseLSTM

def load_lstm_model(model_path, device='cpu'):
    """
    Загружает LSTM модель с весами
    """
    print(f"Загрузка модели из {model_path}...")

    model = EnhancedPoseLSTM(
        input_size=51,
        hidden_size=256,
        num_layers=3,
        num_classes=23,
        dropout=0.4
    )

    # Загружаем веса
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Модель загружена на {device}")
    return model

def classify_person_sequence(segment, lstm_model, preprocessor, class_names, device='cpu'):
    """
    Реальная классификация с вашей LSTM моделью
    """
    keypoints = segment["keypoints"]  # [n_frames, 51] - ОРИГИНАЛЬНЫЕ координаты

    if len(keypoints) == 0:
        return {
            "predicted_class": "no_exercise",
            "confidence": 0.0,
            "class_idx": -1
        }

    processed = preprocessor.preprocess(keypoints)  # [50, 51] - НОРМАЛИЗОВАННЫЕ координаты

    debug = False
    if debug:
        print(f"До нормализации: X [{np.min(keypoints[:, 0::3]):.1f}, {np.max(keypoints[:, 0::3]):.1f}], Y [{np.min(keypoints[:, 1::3]):.1f}, {np.max(keypoints[:, 1::3]):.1f}]")
        print(f"После нормализации: X [{np.min(processed[:, 0::3]):.1f}, {np.max(processed[:, 0::3]):.1f}], Y [{np.min(processed[:, 1::3]):.1f}, {np.max(processed[:, 1::3]):.1f}]")

    # Подготовка тензора
    input_tensor = torch.tensor(processed, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # [1, 50, 51]

    # Предсказание
    with torch.no_grad():
        outputs = lstm_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    return {
        "predicted_class": class_names[predicted_idx.item()],
        "confidence": confidence.item(),
        "class_idx": predicted_idx.item()
    }