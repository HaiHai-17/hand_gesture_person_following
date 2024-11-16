import cv2
import numpy as np
from ultralytics import YOLO
from gesture_recognition import GestureRecognition
from cvfpscalc import CvFpsCalc
import time


class HandTracking:
    def __init__(self):
        self.model = YOLO("yolov5nu.pt")
        self.conf_thres = 0.5
        self.classes = [0]

        # Khởi tạo GestureRecognition
        self.gesture_detector = GestureRecognition(
            "C:/Users/nhh17/OneDrive/Documents/ros_hand_gesture_recognition-main/model/keypoint_classifier/keypoint_classifier_label.csv",
            "C:/Users/nhh17/OneDrive/Documents/ros_hand_gesture_recognition-main/model/keypoint_classifier/keypoint_classifier.tflite"
        )

        self.cv_fps_calc = CvFpsCalc(buffer_len=10)
        self.cap = cv2.VideoCapture(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        # Biến trạng thái theo dõi
        self.tracking = False
        self.gesture_detected = None
        self.gesture_start_time = None
        self.gesture_confirmed = False
        self.frame_skip_count = 0
        self.skip_interval = 1  # Bỏ qua 2 khung hình sau mỗi lần xử lý

        # Thời gian không phát hiện đối tượng
        self.no_detection_start_time = None
        self.detection_timeout = 10

    def reset_tracking(self):
        self.tracking = False
        self.gesture_detected = None
        self.gesture_confirmed = False
        self.frame_skip_count = 0
        self.no_detection_start_time = None

    def handle_gesture(self, gesture):
        if gesture is not None and gesture != self.gesture_detected:
            self.gesture_start_time = time.time()
            self.gesture_detected = gesture
            self.gesture_confirmed = False

        if self.gesture_start_time and time.time() - self.gesture_start_time >= 3:
            self.gesture_confirmed = True

        if self.gesture_confirmed:
            if self.gesture_detected == 'Start' and not self.tracking:
                self.tracking = True
                self.no_detection_start_time = None
            elif self.gesture_detected == 'Stop' and self.tracking:
                self.reset_tracking()

            self.gesture_confirmed = False
            self.gesture_detected = None
            self.gesture_start_time = None

    def track_person(self, frame):
        results = self.model.track(source=frame, conf=self.conf_thres, classes=self.classes, verbose=False)
        status_text = "Tracking"

        if results:
            first_result = results[0]
            boxes = first_result.boxes
            if boxes:
                box = boxes[0]
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()

                if cls == 0:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    self.no_detection_start_time = None
                else:
                    if self.no_detection_start_time is None:
                        self.no_detection_start_time = time.time()
        else:
            if self.no_detection_start_time is None:
                self.no_detection_start_time = time.time()
            elif time.time() - self.no_detection_start_time >= self.detection_timeout:
                status_text = "No Detection"
                self.reset_tracking()
                self.gesture_start_time = None
                self.gesture_confirmed = False
                self.gesture_detected = None

        return status_text, center_x, center_y

    def draw_overlay(self, frame, gesture, center_x, center_y, status_text):
        height, width, _ = frame.shape
        width_left = width // 2 - 100
        width_right = width // 2 + 100

        cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)
        cv2.line(frame, (0, height // 2), (width, height // 2), (255, 0, 0), 2)
        cv2.line(frame, (width_left, 0), (width_left, height), (255, 150, 0), 2)
        cv2.line(frame, (width_right, 0), (width_right, height), (255, 150, 0), 2)
        cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
        cv2.putText(frame, gesture, (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        coord_text = f'X:{center_x}    Y:{center_y}'
        cv2.putText(frame, coord_text, (width // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, status_text, (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        fps = self.cv_fps_calc.get()
        debug_image = self.gesture_detector.draw_fps_info(frame, fps)

        return debug_image

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Không thể đọc được khung hình")
                break

            debug_image, gesture = self.gesture_detector.recognize(frame)

            self.handle_gesture(gesture)

            if self.tracking and self.frame_skip_count % self.skip_interval == 0:
                status_text, center_x, center_y = self.track_person(frame)
            else:
                status_text = "Not Tracking"
                center_x = center_y = 0

            debug_image = self.draw_overlay(frame, gesture, center_x, center_y, status_text)
            cv2.imshow('Person Following', debug_image)

            if gesture is None:
                self.gesture_detected = None

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.frame_skip_count += 1

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = HandTracking()
    tracker.run()
