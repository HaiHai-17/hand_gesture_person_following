import cv2
import numpy as np
from ultralytics import YOLO
from gesture_recognition import GestureRecognition
from cvfpscalc import CvFpsCalc
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import torch  # Add this import

class HandTracking(Node):
    def __init__(self):
        super().__init__('hand_tracking_node')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Set device to GPU if available
        self.model = YOLO("yolov5nu.pt").to(self.device)  # Move YOLO model to device
        
        self.conf_thres = 0.5
        self.classes = [0]

        # Initialize GestureRecognition with device if supported
        self.gesture_detector = GestureRecognition(
            "C:/Users/nhh17/OneDrive/Documents/hand_gesture_person_following/model/keypoint_classifier/keypoint_classifier_label.csv",
            "C:/Users/nhh17/OneDrive/Documents/hand_gesture_person_following/model/keypoint_classifier/keypoint_classifier.tflite",
            device=self.device  # Pass device if GestureRecognition supports it
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

        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.target_id = None  # Add this line to store the target's ID

    def reset_tracking(self):
        self.tracking = False
        self.gesture_detected = None
        self.gesture_confirmed = False
        self.frame_skip_count = 0
        self.no_detection_start_time = None

    def _is_bbox_overlap(self, bbox1, bbox2):
        # Helper function to check if two bounding boxes overlap
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        overlap_area = x_overlap * y_overlap

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        overlap_ratio = overlap_area / float(area1 + area2 - overlap_area)

        return overlap_ratio > 0.1  # Adjust the threshold as needed

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Không thể đọc được khung hình")
                break
            
            # Gesture recognition now also returns hand_bbox
            debug_image, gesture, hand_bbox = self.gesture_detector.recognize(frame)

            # Vẽ
            center_x = center_y = 0
            status_text = "Not Tracking"
            location_text = "Center"
            height, width, _ = frame.shape

            width_left = width // 2 - 150
            width_right = width // 2 + 150
            
            cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)
            # cv2.line(frame, (0, height // 2), (width, height // 2), (255, 0, 0), 2)
            cv2.line(frame, (width_left, 0), (width_left, height), (255, 150, 0), 2)
            cv2.line(frame, (width_right, 0), (width_right, height), (255, 150, 0), 2)
            cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
            cv2.putText(frame, gesture, ((width // 2 ) // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # Bắt đầu xác nhận cử chỉ nếu phát hiện mới
            if gesture is not None and gesture != self.gesture_detected:
                self.gesture_start_time = time.time()
                self.gesture_detected = gesture
                self.gesture_confirmed = False

            # Xác nhận cử chỉ nếu giữ cử chỉ trong 3 giây
            if self.gesture_start_time and time.time() - self.gesture_start_time >= 3:
                self.gesture_confirmed = True

            # Kích hoạt hoặc dừng theo dõi
            if self.gesture_confirmed:
                if self.gesture_detected == 'Start' and not self.tracking:
                    self.tracking = True
                    self.no_detection_start_time = None
                    self.target_id = None  # Reset target_id

                    # Run detection to get target_id of the person associated with the gesture
                    results = self.model.track(source=frame, conf=self.conf_thres, classes=self.classes, verbose=False)
                    if results:
                        first_result = results[0]
                        boxes = first_result.boxes
                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            person_bbox = [x1, y1, x2, y2]
                            if self._is_bbox_overlap(person_bbox, hand_bbox):
                                self.target_id = int(box.id[0].cpu().numpy())
                                break
                    if self.target_id is None:
                        # No matching person found, reset tracking
                        self.tracking = False
                        print("Không tìm thấy đối tượng để theo dõi")
                elif self.gesture_detected == 'Stop' and self.tracking:
                    self.reset_tracking()

                # Reset lại xác nhận cử chỉ sau khi kích hoạt hành động
                self.gesture_confirmed = False
                self.gesture_detected = None
                self.gesture_start_time = None

            # Theo dõi khi được kích hoạt và chỉ thực hiện theo dõi khi đạt đủ số khung hình bỏ qua
            if self.tracking and self.frame_skip_count % self.skip_interval == 0:
                results = self.model.track(
                    source=frame,
                    conf=self.conf_thres,
                    classes=self.classes,
                    verbose=False,
                    device=self.device  # Ensure the device is specified
                )
                status_text = "Tracking"

                if results:
                    first_result = results[0]
                    boxes = first_result.boxes
                    target_box = None
                    for box in boxes:
                        track_id = int(box.id[0].cpu().numpy())
                        if track_id == self.target_id:
                            target_box = box
                            break

                    if target_box:
                        xyxy = target_box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        if width_left < center_x < width_right:
                            location_text = "Center"
                        elif center_x < width_left:
                            location_text = "Left"
                        else:
                            location_text = "Right"

                        # Vẽ điểm ước lượng và tọa độ trung tâm
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                        # Control robot based on center_x
                        twist = Twist()
                        if width_left < center_x < width_right:
                            # Go straight
                            twist.linear.x = 0.5
                            twist.angular.z = 0.0
                        elif center_x < width_left:
                            # Turn left
                            twist.linear.x = 0.0
                            twist.angular.z = 0.5
                        elif center_x > width_right:
                            # Turn right
                            twist.linear.x = 0.0
                            twist.angular.z = -0.5
                        else:
                            # Stop
                            twist.linear.x = 0.0
                            twist.angular.z = 0.0
                        self.cmd_vel_publisher.publish(twist)
                    else:
                        # Target lost, reset tracking
                        print("Mất dấu đối tượng theo dõi")
                        self.reset_tracking()
                else:
                    # No detections
                    status_text = "No Detection"
                    if self.no_detection_start_time is None:
                        self.no_detection_start_time = time.time()
                    elif time.time() - self.no_detection_start_time >= self.detection_timeout:
                        self.reset_tracking()
                        self.gesture_start_time = None
                        self.gesture_confirmed = False
                        self.gesture_detected = None
                    # No detection, stop the robot
                    twist = Twist()
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_vel_publisher.publish(twist)
            else:
                status_text = "Not Tracking"
                # Not tracking, stop the robot
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_publisher.publish(twist)

            # Cập nhật đếm khung hình và thực hiện bỏ qua
            self.frame_skip_count += 1

            # Vẽ thông tin FPS lên hình ảnh
            fps = self.cv_fps_calc.get()
            debug_image = self.gesture_detector.draw_fps_info(frame, fps)
            coord_text = f'X:{center_x}    Y:{center_y}'
            cv2.putText(frame, coord_text, (width // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, status_text, ((width // 2 ) + (width // 4), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, location_text, (10, height), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow('Person Following', debug_image)

            # Reset cử chỉ nếu không có cử chỉ nào được phát hiện trong khung hình
            if gesture is None:
                self.gesture_detected = None

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    tracker = HandTracking()
    try:
        tracker.run()
    except KeyboardInterrupt:
        pass
    tracker.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
