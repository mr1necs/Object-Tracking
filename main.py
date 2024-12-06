import cv2
import numpy as np
from collections import deque
from yolo_model import YOLOModel
from video_stream import VideoStream
from utils import get_arguments
from multiprocessing import current_process, Pool
import logging
import time

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s")

def process_frame_segment(segment_data, model):
    start_time = time.time()
    frame_segment, step = segment_data
    logging.debug(f"Процесс {current_process().name} начал обработку сегмента.")

    results = model.process_frame(frame_segment)
    detected_objects = []

    for r in results:
        detections = r.boxes
        for det in detections:
            conf = det.conf.cpu().numpy()[0]
            cls = int(det.cls.cpu().numpy()[0])
            class_name = model.names(cls)

            if class_name.lower() in {'frisbee', 'sports ball', 'apple', 'orange', 'cake', 'clock'} and conf >= 0.3:
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                detected_objects.append((class_name, conf, (x1 + step, y1, x2 + step, y2)))

    end_time = time.time()
    logging.debug(
        f"Процесс {current_process().name} завершил обработку сегмента. Время обработки: {end_time - start_time:.4f} сек.")
    return detected_objects


def segment_processor(segment_data, model):
    return process_frame_segment(segment_data, model)


def merge_results(results):
    unique_detections = {}
    for segment_results in results:
        for obj in segment_results:
            key = (obj[0], tuple(obj[2]))
            if key not in unique_detections or obj[1] > unique_detections[key][1]:
                unique_detections[key] = obj
    logging.debug(f"Результаты объединены. Всего найдено {len(unique_detections)} объектов.")
    return list(unique_detections.values())


def main():
    # Получаем аргументы из командной строки
    args = get_arguments()

    # Инициализация модели и камеры
    model = YOLOModel(model_path="yolo11n.pt", device=args["device"])
    camera = VideoStream(video_path=args["camera"])
    pts = deque(maxlen=args["buffer"])

    while True:
        grabbed, frame = camera.get_frame()
        if not grabbed:
            logging.warning("Не удалось захватить кадр.")
            break

        height, width = frame.shape[:2]
        segments = [
            (frame[:, part_id * (width // 3):(part_id + 1) * (width // 3)], part_id * (width // 3))
            for part_id in range(3)
        ]

        # Используем Pool для параллельной обработки
        start_time = time.time()  # Время начала обработки кадра

        with Pool(processes=3) as pool:  # Запускаем 3 процесса для обработки сегментов
            results = pool.starmap(segment_processor, [(seg, model) for seg in segments])

        # Объединение результатов
        merged_detections = merge_results(results)

        # Рисование траектории
        for class_name, conf, (x1, y1, x2, y2) in merged_detections:
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            pts.appendleft(center)

        # Отображение траектории
        for i in range(1, len(pts)):
            if pts[i - 1] is not None and pts[i] is not None:
                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # Показать изображение
        cv2.imshow("Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Программа завершена по запросу пользователя.")
            break

        # Время обработки кадра
        end_time = time.time()
        logging.debug(f"Время обработки кадра: {end_time - start_time:.4f} сек.")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
