import cv2
from ultralytics import YOLO
from gesture_recognition import GestureRecognition  # Đảm bảo import đúng
from cvfpscalc import CvFpsCalc
import time

class HandTracking:

    def __init__(self):
        # Tải mô hình YOLOv5n đã được huấn luyện
        self.model = YOLO("/yolo/yolov5n.pt")
        
        # Thiết lập các tham số theo dõi
        self.conf_thres = 0.25
        self.classes = 0
        
        # Khởi tạo GestureRecognition
        self.gesture_detector = GestureRecognition(
            "C:/Users/nhh17/OneDrive/Documents/ros_hand_gesture_recognition-main/model/keypoint_classifier/keypoint_classifier_label.csv",
            "C:/Users/nhh17/OneDrive/Documents/ros_hand_gesture_recognition-main/model/keypoint_classifier/keypoint_classifier.tflite"
        )
        
        self.cv_fps_calc = CvFpsCalc(buffer_len=10)
        self.cap = cv2.VideoCapture(1)  # Sử dụng camera

        # Biến trạng thái theo dõi
        self.tracking = False
        self.gesture_detected = None  # Biến lưu cử chỉ đã nhận diện
        self.start_tracking_time = None  # Biến lưu thời gian bắt đầu theo dõi
        self.gesture_start_time = None  # Biến lưu thời gian bắt đầu nhận diện cử chỉ
        self.gesture_confirmed = False  # Trạng thái xác nhận cử chỉ

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Không thể đọc được khung hình")
                break
            
            # Nhận diện cử chỉ
            debug_image, gesture = self.gesture_detector.recognize(frame)
            print("Detected gesture:", gesture)

            # Bắt đầu xác nhận cử chỉ nếu phát hiện mới
            if gesture is not None and gesture != self.gesture_detected:
                self.gesture_start_time = time.time()
                self.gesture_detected = gesture
                self.gesture_confirmed = False

            # Xác nhận cử chỉ nếu giữ cử chỉ trong 3 giây
            if self.gesture_start_time and time.time() - self.gesture_start_time >= 3:
                self.gesture_confirmed = True  # Cử chỉ đã được xác nhận

            # Nếu cử chỉ được xác nhận và là "start" hoặc "stop"
            if self.gesture_confirmed:
                if self.gesture_detected == 'Start' and not self.tracking:
                    self.start_tracking_time = time.time()  # Ghi lại thời gian để delay trước theo dõi
                    self.tracking = True
                elif self.gesture_detected == 'Stop' and self.tracking:
                    self.tracking = False  # Dừng theo dõi
                    self.start_tracking_time = None  # Reset thời gian bắt đầu theo dõi

                # Reset lại xác nhận cử chỉ sau khi kích hoạt hành động
                self.gesture_confirmed = False
                self.gesture_detected = None
                self.gesture_start_time = None

            # Chờ thời gian delay 5 giây trước khi thực hiện theo dõi
            
            if self.tracking:  # Nếu đang ở trạng thái theo dõi
                results = self.model.track(source=frame, conf=self.conf_thres, classes=self.classes)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        
                        # Vẽ bounding box, nhãn và độ tin cậy
                        if cls == 0:  # Chỉ theo dõi lớp 'person'
                            x1, y1, x2, y2 = map(int, xyxy)
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)

                            # Vẽ point và tọa độ trung tâm
                            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                            coord_text = f'({center_x}, {center_y})'
                            print('Point: ', coord_text)
                            cv2.putText(frame, coord_text, (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Vẽ thông tin FPS lên hình ảnh
            fps = self.cv_fps_calc.get()
            debug_image = self.gesture_detector.draw_fps_info(frame, fps)

            cv2.imshow('Kết quả theo dõi', debug_image)

            # Reset cử chỉ nếu không có cử chỉ nào được phát hiện trong khung hình
            if gesture is None:
                self.gesture_detected = None

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = HandTracking()
    tracker.run()
