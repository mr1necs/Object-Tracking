from model_yolo import YOLOModel
from video_processor import VideoProcessor
from object_tracker import ObjectTracker
from utils import get_arguments
import cv2


class MainApp:
    """
    Основной класс приложения, который связывает все компоненты.
    """

    def __init__(self, args):
        """
        Инициализация основных компонентов приложения.
        :param args: Аргументы командной строки.
        """
        self.model = YOLOModel('yolo11n.pt')
        self.model.select_device(args["device"])
        self.video_processor = VideoProcessor(args["camera"])
        self.tracker = ObjectTracker(args["buffer"], args["overlay"], args["timeout"])

    def run(self):
        """
        Основной цикл обработки кадров. Выполняет захват кадров, обработку ROI или полного изображения,
        отрисовку объектов и траектории.
        """
        while True:
            grabbed, frame = self.video_processor.get_frame()
            if not grabbed:
                break

            height, width = frame.shape[:2]

            # Если объект потерян, выполняем поиск на всем кадре
            if self.tracker.tracking_roi is None or self.tracker.lost_counter >= self.tracker.timeout:
                results = self.tracker.process_full_frame(self.model, frame, height, width)
                self.tracker.update_tracking_roi(results)
            else:
                # Если объект отслеживается, обрабатываем только ROI
                frame = self.tracker.process_roi(self.model, frame)

            # Отрисовка траектории
            self.tracker.draw_trajectory(frame)

            # Отображение результата
            cv2.imshow("Object Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_processor.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_arguments()
    app = MainApp(args)
    app.run()
