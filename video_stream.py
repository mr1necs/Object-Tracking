import cv2
import imutils
import logging

class VideoStream:
    """
    Класс для управления видеопотоком. Поддерживает открытие, чтение кадров и освобождение видеопотока.
    """

    def __init__(self, video_path=None):
        """
        Инициализация видеопотока.

        :param video_path: Путь к видеофайлу. Если None, используется камера по умолчанию.
        """
        self.camera = self.get_video(video_path)

    def get_video(self, video_path):
        """
        Открытие видеопотока.

        :param video_path: Путь к видеофайлу. Если None, используется камера по умолчанию.
        :return: Объект видеопотока cv2.VideoCapture.
        """
        camera = cv2.VideoCapture(0 if video_path is None else video_path)
        if not camera.isOpened():
            logging.error("Не удалось открыть видеопоток.")
            exit(1)
        return camera

    def get_frame(self):
        """
        Чтение следующего кадра из видеопотока.

        :return: Кортеж (grabbed, frame), где grabbed — успешность захвата кадра,
                 frame — масштабированный кадр или None, если кадр не захвачен.
        """
        grabbed, frame = self.camera.read()
        if not grabbed:
            logging.info("Видеопоток завершён или кадр не захвачен.")
            return grabbed, None
        return grabbed, imutils.resize(frame, width=800)

    def release(self):
        """
        Освобождение видеопотока.
        """
        if self.camera.isOpened():
            self.camera.release()
            logging.info("Видеопоток освобождён.")
