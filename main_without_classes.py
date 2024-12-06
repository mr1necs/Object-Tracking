from argparse import ArgumentParser
from ultralytics import YOLO
from collections import deque
from torch.backends import mps
from torch import cuda
import numpy as np
import cv2
import imutils
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial  # Импортируем functools.partial

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def get_arguments():
    ap = ArgumentParser()
    ap.add_argument("-d", "--device", type=str, default='cpu', help="device: 'mps', 'cuda' or 'cpu'")
    ap.add_argument("-c", "--camera", type=str, default=None, help="path to the optional video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="maximum buffer size for trajectory")
    return vars(ap.parse_args())


def get_model():
    args = get_arguments()
    device = args["device"]

    model = YOLO('yolo11n.pt')

    device = (
        'mps' if device == 'mps' and mps.is_available() else
        'cuda' if device == 'cuda' and cuda.is_available() else
        'cpu'
    )

    model.to(device)
    return model


def get_video(video_path=None):
    camera = cv2.VideoCapture(0 if not video_path else video_path)

    if not camera.isOpened():
        print("Ошибка при открытии видеопотока.")
        exit()
    return camera


def process_frame_segment(segment_data, model):
    classes = {'frisbee', 'sports ball', 'apple', 'orange', 'cake', 'clock'}

    frame_segment, y, x = segment_data
    results = model(frame_segment)
    detected_objects = []

    for r in results:
        detections = r.boxes
        for det in detections:
            xyxy = det.xyxy[0].cpu().numpy()
            conf = det.conf.cpu().numpy()[0]
            cls = int(det.cls.cpu().numpy()[0])
            class_name = model.names[cls]

            if class_name.lower() in classes and conf >= 0.3:
                x1, y1, x2, y2 = xyxy
                detected_objects.append((class_name, conf, (x1 + x, y1 + y, x2 + x, y2 + y)))
                print(x, y)
                print((x1, y1, x2, y2))
                print((x1 + x, y1 + y, x2 + x, y2 + y))
    return detected_objects


def merge_results(results):
    unique_detections = {}
    for segment_results in results:
        for obj in segment_results:
            key = (obj[0], tuple(obj[2]))
            if key not in unique_detections or obj[1] > unique_detections[key][1]:
                unique_detections[key] = obj
    return list(unique_detections.values())


def check_intersection(obj1, obj2):
    """Проверка на пересечение двух объектов."""
    x1_1, y1_1, x2_1, y2_1 = obj1[2]
    x1_2, y1_2, x2_2, y2_2 = obj2[2]

    # Проверяем, пересекаются ли два прямоугольника
    return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)


def choose_larger_radius(objects):
    """Выбираем объект с большим радиусом, если они пересекаются."""
    if len(objects) <= 1:
        return objects

    chosen_object = objects[0]
    for obj in objects[1:]:
        x1, y1, x2, y2 = chosen_object[2]
        r1 = abs(y2 - y1) / 2  # Радиус первого объекта
        x1, y1, x2, y2 = obj[2]
        r2 = abs(y2 - y1) / 2  # Радиус второго объекта

        # Если пересекаются, выбираем объект с большим радиусом
        if check_intersection(chosen_object, obj):
            if r2 > r1:
                chosen_object = obj

    return [chosen_object]  # Возвращаем только один объект


def main():
    args = get_arguments()
    camera = get_video(args["camera"])
    pts = deque(maxlen=args["buffer"])

    model = get_model()  # Загружаем модель один раз

    # Используем partial для создания функции, которая включает модель
    process_with_model = partial(process_frame_segment, model=model)

    with Pool(processes=cpu_count()) as pool:
        while True:
            grabbed, frame = camera.read()
            if not grabbed:
                break

            frame = imutils.resize(frame, width=800)
            height, width = frame.shape[:2]
            overlay = 50

            segments = [
                (0, 0, (height // 2 + overlay), (width // 2 + overlay)),
                (0, (width // 2 - overlay), (height // 2 + overlay), width),
                ((height // 2 - overlay), 0, height, (width // 2 + overlay)),
                ((height // 2 - overlay), (width // 2 - overlay), height, width)
            ]

            results = pool.map(process_with_model, [(frame[x1:x2, y1:y2], x1, y1) for (x1, y1, x2, y2) in segments])
            merged_detections = merge_results(results)

            # Выбираем один объект, если они пересекаются
            chosen_detection = choose_larger_radius(merged_detections)

            for class_name, conf, (x1, y1, x2, y2) in chosen_detection:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                center = (center_x, center_y)
                radius = int(abs((y1 - y2) / 2))
                pts.appendleft(center)
                cv2.circle(frame, center, radius, (0, 255, 255), 3)

            for i in range(1, len(pts)):
                if pts[i - 1] is not None and pts[i] is not None:
                    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

            cv2.imshow("Object Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
