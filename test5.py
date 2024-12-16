import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
from gesture_recognition import GestureRecognition  # Đảm bảo import đúng

# Khởi tạo model YOLO
model = YOLO('yolov5n.pt')  # Sử dụng YOLOv5n model nhẹ

# Khởi tạo DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, max_iou_distance=0.7)

# Màu sắc cho các trạng thái
RED_COLOR = (0, 0, 255)
YELLOW_COLOR = (0, 255, 255)
GREEN_COLOR = (0, 255, 0)
BLACK_COLOR = (0, 0, 0)

# Khởi tạo Mediapipe cho nhận diện bàn tay
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo mô hình nhận diện cử chỉ tay
gesture_detector = GestureRecognition("C:/Users/nhh17/OneDrive/Documents/hand_gesture_person_following/model/keypoint_classifier/keypoint_classifier_label.csv",
                                     "C:/Users/nhh17/OneDrive/Documents/hand_gesture_person_following/model/keypoint_classifier/keypoint_classifier.tflite")

# Mở camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

# Hàm xác định điểm trung tâm của bounding box
def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

# Hàm kiểm tra xem một điểm có nằm trong một bounding box không
def is_point_in_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

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

    # Nhận diện bàn tay và kiểm tra xem có nằm trong box người
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Khởi tạo debug_image từ frame gốc
    debug_image = frame.copy()

    # Kiểm tra các bàn tay
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Xác định bounding box của bàn tay
            x_min = min([landmark.x for landmark in landmarks.landmark]) * frame.shape[1]
            y_min = min([landmark.y for landmark in landmarks.landmark]) * frame.shape[0]
            x_max = max([landmark.x for landmark in landmarks.landmark]) * frame.shape[1]
            y_max = max([landmark.y for landmark in landmarks.landmark]) * frame.shape[0]

            hand_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))

            # Kiểm tra xem các điểm của bàn tay có nằm trong bounding box của người không
            for track in tracks:
                if track.is_confirmed() and track.time_since_update <= 1:
                    person_bbox = track.to_ltrb()
                    # Kiểm tra từng điểm landmark của bàn tay xem có nằm trong bounding box của người không
                    hand_in_person = False
                    for landmark in landmarks.landmark:
                        point = (landmark.x * frame.shape[1], landmark.y * frame.shape[0])
                        if is_point_in_bbox(point, person_bbox):
                            hand_in_person = True
                            break

                    if hand_in_person:
                        print(f"Hand in box person - ID: {track.track_id}")
                        break

            # Nhận diện cử chỉ tay và hiển thị trên debug_image
            debug_image, gesture = gesture_detector.recognize(frame)

    else:
        # Nếu không có bàn tay, nhận diện cử chỉ tay với frame gốc
        debug_image, gesture = gesture_detector.recognize(frame)

    # Vẽ các bounding box của YOLO và DeepSORT trên debug_image
    for track in tracks:
        if track.is_confirmed() and track.time_since_update <= 1:
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            track_id = track.track_id

            # Vẽ bounding box và nhãn của đối tượng
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), RED_COLOR, 2)
            font_scale = 0.5
            thickness = 1
            text_size, _ = cv2.getTextSize(f"Person - {track_id}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_width, text_height = text_size

            # Tạo nền cho nhãn
            cv2.rectangle(debug_image, (x1, y1 - text_height - 5), (x1 + text_width, y1), RED_COLOR, -1)

            # Vẽ chữ
            cv2.putText(debug_image, f"Person - {track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Hiển thị khung hình nhận diện cử chỉ tay và bounding box
    cv2.imshow('Gesture Recognition', debug_image)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()