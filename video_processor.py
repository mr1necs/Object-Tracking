import cv2
import imutils


class VideoProcessor:
    """
    Класс для захвата видео и предобработки кадров.
    """

    def __init__(self, video_path):
        """
        Инициализация видеопотока.
        :param video_path: Путь к видеофайлу или None для использования веб-камеры.
        """
        self.camera = cv2.VideoCapture(0 if video_path is None else video_path)
        if not self.camera.isOpened():
            raise ValueError("Ошибка при открытии видеопотока.")

    def get_frame(self):
        """
        Захватывает следующий кадр из видеопотока и изменяет его размер.
        :return: Флаг захвата и кадр.
        """
        grabbed, frame = self.camera.read()
        if grabbed:
            frame = imutils.resize(frame, width=800)
        return grabbed, frame

    def release(self):
        """
        Освобождает видеопоток.
        """
        self.camera.release()
