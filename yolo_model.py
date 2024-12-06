from ultralytics import YOLO
from torch.backends import mps
from torch import cuda
import logging

class YOLOModel:
    """
    Обёртка для модели YOLO, обеспечивающая выбор устройства, загрузку и детектирование объектов.
    """
    def __init__(self, model_path, device='cpu'):
        """
        Инициализация модели YOLOModel.

        :param model_path: Путь к файлу модели YOLO.
        :param device: Предпочитаемое устройство ('cpu', 'cuda', 'mps').
        """

        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        self.device = self.select_device(device)
        self.model = self.load_model(model_path)
        self.classes = {'frisbee', 'sports ball', 'apple', 'orange', 'cake', 'clock'}
        self.conf = 0.3

    def load_model(self, model_path):
        """
        Загрузка модели YOLO на выбранное устройство.

        :param model_path: Путь к файлу модели YOLO.
        :return: Загруженная модель YOLO.
        """
        try:
            model = YOLO(model_path).to(self.device)
            logging.info("Модель успешно загружена.")
            return model

        except Exception as e:
            logging.error(f"Ошибка при загрузке модели YOLO: {e}")
            exit(1)

    def select_device(self, device_preference='cpu'):
        """
        Выбор устройства для вычислений.

        :param device_preference: Предпочитаемое устройство ('cpu', 'cuda', 'mps').
        :return: Доступное устройство в виде строки.
        """
        device = (
            'mps' if device_preference == 'mps' and mps.is_available() else
            'cuda' if device_preference == 'cuda' and cuda.is_available() else
            'cpu'
        )
        if device != device_preference:
            logging.warning(f"Предпочитаемое устройство '{device_preference}' недоступно.")
        logging.info(f"Выбранное устройство: {device}")
        return device

    def process_frame(self, frame):
        """
        Обработка кадра и детектирование объектов.

        :param frame: Входной кадр (изображение).
        :return: Список детектированных объектов [(название класса, уверенность, координаты границ)].
        """
        return self.model(frame)

    def names(self, cls):
        return self.model.names[cls]
