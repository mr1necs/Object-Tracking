import cv2
import numpy as np
from collections import deque

class ObjectTracker:
    def __init__(self, buffer_size=64, roi_size=200, timeout=30):
        """
        Инициализация трекера объекта.

        :param buffer_size: Размер буфера для хранения координат траектории.
        :param roi_size: Размер области интереса (ROI) для локальной детекции.
        :param timeout: Количество кадров без обнаружения до возврата к глобальной детекции.
        """
        self.tracks = deque(maxlen=buffer_size)
        self.roi_size = roi_size
        self.timeout = timeout
        self.missed_frames = 0
        self.last_position = (0, 0)  # Последняя известная позиция объекта

    def update_tracks(self, detections):
        """
        Обновляет траектории на основе новых детекций.

        :param detections: Список детекций, содержащих координаты боксов.
        """
        if detections:
            detection = detections[0]  # Берем первую детекцию (наиболее уверенную)
            x_center = int((detection['x1'] + detection['x2']) / 2)
            y_center = int((detection['y1'] + detection['y2']) / 2)

            self.tracks.appendleft((x_center, y_center))
            self.last_position = (x_center, y_center)
            self.missed_frames = 0  # Сбрасываем счетчик пропущенных кадров
        else:
            self.missed_frames += 1

    def process_full_frame(self, model, frame):
        """
        Выполняет глобальную детекцию на всем кадре.

        :param model: YOLO модель для детекции объектов.
        :param frame: Полный кадр для анализа.
        :return: Обновленный кадр.
        """
        results = model.detect(frame)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': cls, 'conf': conf})

        self.update_tracks(detections)
        return self.visualize(frame, self.tracks)

    def process_roi(self, model, frame):
        """
        Выполняет локальную детекцию на ROI (Region of Interest).

        :param model: YOLO модель для детекции объектов.
        :param frame: Полный кадр для анализа.
        :return: Обновленный кадр.
        """
        if self.missed_frames > self.timeout:
            print("Switching to full frame detection due to timeout.")
            return self.process_full_frame(model, frame)

        center_x, center_y = self.last_position
        roi_x1 = max(0, int(center_x - self.roi_size // 2))
        roi_x2 = min(frame.shape[1], int(center_x + self.roi_size // 2))
        roi_y1 = max(0, int(center_y - self.roi_size // 2))
        roi_y2 = min(frame.shape[0], int(center_y + self.roi_size // 2))

        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi_frame.size == 0:
            print("Invalid ROI, switching to full frame detection.")
            return self.process_full_frame(model, frame)

        results = model.detect(roi_frame)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({'x1': x1 + roi_x1, 'y1': y1 + roi_y1,
                                   'x2': x2 + roi_x1, 'y2': y2 + roi_y1,
                                   'class': cls, 'conf': conf})

        self.update_tracks(detections)
        return self.visualize(frame, self.tracks)

    def visualize(self, frame, tracks):
        """
        Визуализирует треки и область интереса на кадре.

        :param frame: Кадр для визуализации.
        :param tracks: Координаты треков.
        :return: Кадр с наложенными визуализациями.
        """
        for i in range(1, len(tracks)):
            if tracks[i - 1] is None or tracks[i] is None:
                continue
            cv2.line(frame, tracks[i - 1], tracks[i], (0, 255, 0), 2)

        if self.missed_frames <= self.timeout:
            center_x, center_y = self.last_position
            roi_x1 = max(0, int(center_x - self.roi_size // 2))
            roi_x2 = min(frame.shape[1], int(center_x + self.roi_size // 2))
            roi_y1 = max(0, int(center_y - self.roi_size // 2))
            roi_y2 = min(frame.shape[0], int(center_y + self.roi_size // 2))

            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

        return frame

    def draw_trajectory(self, frame):
        for i in range(1, len(self.tracks)):
            if self.tracks[i - 1] is None or self.tracks[i] is None:
                continue
            cv2.line(frame, self.tracks[i - 1], self.tracks[i], (0, 255, 0), 2)

    def update_tracking_roi(self, detections):
        """
        Обновляет область интереса (ROI) на основе новых детекций.

        :param detections: Результаты детекции (список или массив с координатами боксов).
        """
        # Проверяем, что detections не пустой
        if detections is not None and len(detections) > 0:  # Для списков
            # Берем первую детекцию как основную
            detection = detections[0]
            x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']

            # Обновляем последнюю позицию центра ROI
            self.last_position = (
                (x1 + x2) // 2,  # Центр по X
                (y1 + y2) // 2  # Центр по Y
            )
            self.missed_frames = 0  # Сбрасываем счетчик пропущенных кадров

        elif isinstance(detections, np.ndarray) and detections.size > 0:  # Для массивов NumPy
            detection = detections[0]  # Берем первую детекцию
            x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]

            self.last_position = (
                (x1 + x2) // 2,
                (y1 + y2) // 2
            )
            self.missed_frames = 0

        else:
            # Если детекций нет, увеличиваем счетчик пропущенных кадров
            self.missed_frames += 1



