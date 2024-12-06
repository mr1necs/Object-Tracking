import logging
from argparse import ArgumentParser
from ultralytics import YOLO
from collections import deque
from torch.backends import mps
from torch import cuda
import numpy as np
import cv2
import imutils

logging.basicConfig(level=logging.INFO)

def get_arguments():
    ap = ArgumentParser()
    ap.add_argument("-d", "--device", type=str, default='cpu', help="device: 'mps', 'cuda' or 'cpu'")
    ap.add_argument("-c", "--camera", type=str, default=None, help="path to the optional video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="maximum buffer size for trajectory")
    return vars(ap.parse_args())

def get_model(device):
    try:
        model = YOLO('yolo11n.pt')
        logging.info(f"Model loaded on {device}.")
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        exit()
    model.to(device)
    return model

def get_device(device_arg):
    device = (
        'mps' if device_arg == 'mps' and mps.is_available() else
        'cuda' if device_arg == 'cuda' and cuda.is_available() else
        'cpu'
    )
    if device_arg not in ['mps', 'cuda', 'cpu']:
        logging.warning(f"Invalid device argument '{device_arg}'. Defaulting to 'cpu'.")
    logging.info(f"Using device: {device}")
    return device

def get_video(video_path=None):
    camera = cv2.VideoCapture(0 if not video_path else video_path)
    if not camera.isOpened():
        logging.error("Error opening video stream.")
        exit()
    return camera

def process_frame(frame, model):
    classes = {'frisbee', 'sports ball', 'apple', 'orange', 'cake', 'clock'}

    results = model(frame)
    detected_objects = []

    for r in results:
        detections = r.boxes
        for det in detections:
            xyxy = det.xyxy[0].cpu().numpy()
            conf = det.conf.cpu().numpy()[0]
            cls = int(det.cls.cpu().numpy()[0])
            class_name = model.names[cls]

            if class_name.lower() in classes and conf >= 0.3:
                detected_objects.append((conf, xyxy))
    return detected_objects

def main():
    args = get_arguments()
    device = get_device(args["device"])
    camera = get_video(args["camera"])
    pts = deque(maxlen=args["buffer"])
    model = get_model(device)

    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            logging.info("End of video stream or no frame grabbed.")
            break

        frame = imutils.resize(frame, width=800)
        detected_objects = process_frame(frame, model)

        for conf, (x1, y1, x2, y2) in detected_objects:
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
            logging.info("User requested exit.")
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()