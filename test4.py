import cv2
import time
from ultralytics import YOLO
import mediapipe as mp
from deep_sort_realtime.deepsort_tracker import DeepSort
from gesture_recognition import GestureRecognition

# Khởi tạo model YOLO
model = YOLO('yolov5n.pt')  # Sử dụng YOLOv5n model nhẹ

# Khởi tạo DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, max_iou_distance=0.7)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Màu sắc
RED_COLOR = (0, 0, 255)
YELLOW_COLOR = (0, 255, 255)
GREEN_COLOR = (0, 255, 0)

# Khởi tạo GestureRecognition
gesture_detector = GestureRecognition("C:/Users/nhh17/OneDrive/Documents/hand_gesture_person_following/model/keypoint_classifier/keypoint_classifier_label.csv",
                                     "C:/Users/nhh17/OneDrive/Documents/hand_gesture_person_following/model/keypoint_classifier/keypoint_classifier.tflite")

# Mở camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

# Biến theo dõi cử chỉ và trạng thái
gesture_start_time = None
gesture_confirmed = False
gesture_detected = None  # Khởi tạo giá trị mặc định cho gesture_detected
tracking = False
no_detection_start_time = None
current_state = "None"  # Trạng thái ban đầu là "None"

# Hàm xác định điểm trung tâm của bounding box
def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

# Hàm vẽ khung xương (pose skeleton)
def draw_pose_skeleton(frame, pose_landmarks):
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = pose_landmarks.landmark[start_idx]
        end_point = pose_landmarks.landmark[end_idx]

        start_x, start_y = int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0])
        end_x, end_y = int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0])
        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

# Hàm cập nhật trạng thái dựa trên cử chỉ và xác nhận cử chỉ trong 3 giây
def update_state_based_on_gesture(gesture):
    global gesture_start_time, gesture_confirmed, gesture_detected, current_state

    # Nếu cử chỉ mới, bắt đầu xác nhận
    if gesture != gesture_detected:
        gesture_start_time = time.time()  # Lưu thời gian bắt đầu
        gesture_confirmed = False
        gesture_detected = gesture  # Cập nhật cử chỉ

    # Kiểm tra thời gian để xác nhận cử chỉ
    if gesture_start_time and time.time() - gesture_start_time >= 3:
        gesture_confirmed = True

    # Cập nhật trạng thái khi cử chỉ đã được xác nhận
    if gesture_confirmed:
        if gesture == "Ready":
            current_state = "Ready"
        elif gesture == "Follow" and current_state == "Ready":
            current_state = "Follow"
        elif gesture == "Stop":
            current_state = "Ready"
            # Nếu Stop, hủy theo dõi
            reset_tracking()

# Hàm khôi phục trạng thái theo dõi
def reset_tracking():
    global tracking, no_detection_start_time
    tracking = False
    no_detection_start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc dữ liệu từ camera.")
        break

    # Dự đoán các bounding box (hộp giới hạn) với YOLOv5
    yolo_results = model.predict(frame, conf=0.5)[0]
    detections = []
    person_bboxes = []

    for box in yolo_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ hộp
        conf = float(box.conf[0])  # Độ tin cậy
        cls = int(box.cls[0])  # Class ID

        if cls == 0:  # Chỉ xử lý class "person" (class 0)
            detections.append(([x1, y1, x2, y2], conf, cls))
            person_bboxes.append([x1, y1, x2, y2])

    # Cập nhật và theo dõi đối tượng với DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id  # ID theo dõi
        ltrb = track.to_ltrb()  # Tọa độ hộp: left, top, right, bottom
        x1, y1, x2, y2 = map(int, ltrb)

        # Nhận diện cử chỉ bàn tay
        debug_image, gesture = gesture_detector.recognize(frame)
        print("Detected gesture:", gesture)

        # Cập nhật trạng thái theo cử chỉ và kiểm tra xác nhận
        update_state_based_on_gesture(gesture)

        # Xử lý màu sắc và hành động dựa trên trạng thái
        if current_state == "Ready":
            color = YELLOW_COLOR
            draw_skeleton = False
        elif current_state == "Follow":
            color = GREEN_COLOR
            draw_skeleton = True
        elif current_state == "Stop":
            color = RED_COLOR
            draw_skeleton = False
        else:
            color = RED_COLOR  # Trạng thái ban đầu là màu đỏ
            draw_skeleton = False

        # Vẽ bounding box của người với màu đã cập nhật
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Vẽ điểm trung tâm của bounding box nếu đang ở trạng thái Follow
        if current_state == "Follow":
            center_x, center_y = get_center_of_bbox([x1, y1, x2, y2])
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Vẽ điểm trung tâm

            # Vẽ khung xương (pose skeleton) nếu đang ở trạng thái "Follow"
            if draw_skeleton:
                pose_results = pose.process(frame)
                if pose_results.pose_landmarks:
                    draw_pose_skeleton(frame, pose_results.pose_landmarks)

        # Kích thước chữ và độ dày
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(f"Person - {track_id}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_width, text_height = text_size

        # Tạo nền cho nhãn
        cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)

        # Vẽ chữ
        cv2.putText(frame, f"Person - {track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Hiển thị khung hình
    cv2.imshow("Person and Hand Gesture Recognition", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
