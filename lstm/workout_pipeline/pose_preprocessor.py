import numpy as np

class PosePreprocessor:
    """Класс для предобработки последовательностей ключевых точек"""
    def __init__(self, seq_length=50, training=False):
        self.seq_length = seq_length
        self.training = training

    def random_drop_frames(self, sequence, target_length):
        """
        Удаляет случайные кадры из последовательности (имитация потерь)
        sequence: [N, 51] - исходная последовательность
        target_length: нужная длина
        """
        if len(sequence) <= target_length:
            return sequence

        # Случайно выбираем какие кадры сохранить
        indices_to_keep = np.random.choice(
            len(sequence),
            size=target_length,
            replace=False
        )
        indices_to_keep.sort()

        return sequence[indices_to_keep]

    def normalize_pose(self, sequence):
        """Нормализует координаты относительно таза"""
        if len(sequence) == 0:
            return sequence

        seq_reshaped = sequence.reshape(-1, 17, 3)

        for i in range(len(seq_reshaped)):
            frame = seq_reshaped[i]

            # Находим центр таза (точки 11, 12)
            if np.all(frame[11, 2] > 0.1) and np.all(frame[12, 2] > 0.1):
                pelvis_center = (frame[11, :2] + frame[12, :2]) / 2
            else:
                # Используем среднее всех видимых точек
                visible_points = frame[frame[:, 2] > 0.1]
                if len(visible_points) > 0:
                    pelvis_center = np.mean(visible_points[:, :2], axis=0)
                else:
                    continue

            # Центрируем все точки
            seq_reshaped[i, :, :2] = seq_reshaped[i, :, :2] - pelvis_center

        return seq_reshaped.reshape(-1, 51)

    def pad_or_truncate(self, sequence):
        """Приводит последовательность к фиксированной длине self.seq_length"""
        if len(sequence) > self.seq_length:
            # Вместо обрезки конца - удаляем случайные кадры
            sequence = self.random_drop_frames(sequence, self.seq_length)
        else:
            # Дополняем нулями
            padded = np.zeros((self.seq_length, 51))
            padded[:len(sequence)] = sequence
            sequence = padded

        return sequence

    def preprocess(self, keypoints_sequence):
        """
        Полный пайплайн предобработки для одного сегмента
        keypoints_sequence: [n_frames, 51]
        возвращает: [seq_length, 51]
        """
        keypoints_sequence_copy = keypoints_sequence.copy()
        # 1. Нормализация
        normalized = self.normalize_pose(keypoints_sequence_copy)

        # 2. Приведение к фиксированной длине (60 кадров)
        processed = self.pad_or_truncate(normalized)

        return processed
