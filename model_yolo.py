from ultralytics import YOLO
from torch.backends import mps
from torch import cuda
import logging


class YOLOModel:
    """
    Класс для управления моделью YOLO.
    """

    def __init__(self, model_path):
        """
        Инициализация модели YOLO.
        :param model_path: Путь к весам модели.
        """
        # Установка уровня логирования
        logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

        self.model = YOLO(model_path)
        self.device = 'cpu'

    def select_device(self, device_preference='cpu'):
        """
        Устанавливает устройство для работы с моделью и перемещает модель на это устройство.
        :param device_preference: 'cpu', 'cuda', или 'mps'
        """

        self.device = (
            'mps' if device_preference == 'mps' and mps.is_available() else
            'cuda' if device_preference == 'cuda' and cuda.is_available() else
            'cpu'
        )
        self.model.to(self.device)

    def detect(self, frame):
        """
        Выполняет детекцию объектов на переданном кадре.
        :param frame: Входное изображение или сегмент.
        :return: Результаты детекции.
        """

        return self.model(frame)
