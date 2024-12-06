from collections import deque
import numpy as np
import cv2


class ObjectTracker:
    """
    Класс для управления логикой трекинга объектов.
    """

    def __init__(self, buffer_size, overlay, timeout):
        """
        Инициализация трекера.
        :param buffer_size: Максимальный размер очереди для траектории.
        :param overlay: Размер дополнительной области вокруг ROI.
        :param timeout: Количество кадров до перехода к полному поиску.
        """
        self.pts = deque(maxlen=buffer_size)
        self.overlay = overlay
        self.timeout = timeout
        self.lost_counter = 0
        self.tracking_roi = None
        self.classes = {'frisbee', 'sports ball', 'apple', 'orange', 'cake', 'clock'}

    def process_detections(self, results, x_offset=0, y_offset=0):
        """
        Обрабатывает результаты детекции, возвращая список объектов.
        :param results: Результаты детекции модели YOLO.
        :param x_offset: Сдвиг по X для глобальных координат.
        :param y_offset: Сдвиг по Y для глобальных координат.
        :return: Список обнаруженных объектов.
        """
        detected_objects = []
        for r in results:
            detections = r.boxes
            for det in detections:
                xyxy = det.xyxy[0].cpu().numpy()
                conf = det.conf.cpu().numpy()[0]
                cls = int(det.cls.cpu().numpy()[0])
                class_name = r.names[cls]

                if class_name.lower() in self.classes and conf >= 0.3:
                    x1, y1, x2, y2 = xyxy
                    detected_objects.append((class_name, conf, (x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset)))
        return detected_objects

    def update_tracking_roi(self, detections):
        """
        Обновляет текущий ROI для отслеживания объекта.
        :param detections: Список обнаруженных объектов.
        """
        if detections:
            self.lost_counter = 0
            class_name, conf, (x1, y1, x2, y2) = detections[0]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            self.tracking_roi = (center_x, center_y)
        else:
            self.lost_counter += 1
            if self.lost_counter >= self.timeout:
                self.tracking_roi = None

    def process_full_frame(self, model, frame, height, width):
        """
        Выполняет поиск объекта на всем кадре.
        :param model: YOLOModel для выполнения детекции.
        :param frame: Кадр для обработки.
        :param height: Высота кадра.
        :param width: Ширина кадра.
        :return: Обнаруженные объекты.
        """
        overlay = self.overlay
        segments = [
            (frame[0:height // 2 + overlay, 0:width // 2 + overlay], 0, 0),
            (frame[0:height // 2 + overlay, width // 2 - overlay:width], 0, width // 2 - overlay),
            (frame[height // 2 - overlay:height, 0:width // 2 + overlay], height // 2 - overlay, 0),
            (frame[height // 2 - overlay:height, width // 2 - overlay:width], height // 2 - overlay,
             width // 2 - overlay)
        ]

        detected_objects = []
        for segment, y_offset, x_offset in segments:
            results = model.detect(segment)
            detected_objects.extend(self.process_detections(results, x_offset, y_offset))

        return detected_objects

    def process_roi(self, model, frame):
        """
        Обрабатывает кадр, ограниченный текущим ROI.
        :param model: YOLOModel для выполнения детекции.
        :param frame: Кадр для обработки.
        :return: Кадр с обновленным ROI.
        """
        if self.tracking_roi:
            center_x, center_y = self.tracking_roi
            height, width = frame.shape[:2]
            roi_x1 = max(0, center_x - (width // 4) - self.overlay)
            roi_y1 = max(0, center_y - (height // 4) - self.overlay)
            roi_x2 = min(width, center_x + (width // 4) + self.overlay)
            roi_y2 = min(height, center_y + (height // 4) + self.overlay)

            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            detections = self.process_detections(model.detect(roi_frame), roi_y1, roi_x1)

            self.update_tracking_roi(detections)
            return frame

    def draw_trajectory(self, frame):
        """
        Отрисовывает траекторию объекта на кадре.
        :param frame: Кадр для отрисовки.
        """
        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is not None and self.pts[i] is not None:
                thickness = int(np.sqrt(len(self.pts) / float(i + 1)) * 2.5)
                cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)
