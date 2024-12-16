import cv2
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp

# Khởi tạo mô hình YOLOv5
model = YOLO('yolov5n.pt')  # Sử dụng mô hình YOLOv5 nhỏ

# Khởi tạo Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1, enable_segmentation=False, refine_face_landmarks=True)

def draw_hand_skeleton(frame, results):
    mp_drawing.draw_landmarks(
        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

    mp_drawing.draw_landmarks(
        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

# def draw_body_skeleton(frame, results):
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#             mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
#             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

def is_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return thumb_tip.y > index_tip.y

def is_open_hand(hand_landmarks):
    fingers = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    ]
    return all(finger.y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y for finger in fingers)

def is_hand_in_person_area(hand_landmarks, person_box):
    x1, y1, x2, y2 = person_box
    hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1]
    hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0]
    return x1 < hand_x < x2 and y1 < hand_y < y2

def get_person_id_by_hand(tracks, hand_landmarks):
    for track in tracks:
        if track.is_confirmed() and track.get_det_class() == 0:  # Lớp 0 là 'person'
            person_box = track.to_ltrb()
            if is_hand_in_person_area(hand_landmarks, person_box):
                return track.track_id
    return None

def is_person(detection):
    return detection['class'] == 0  # Lớp 0 trong YOLOv5 là 'person'

# Khởi tạo DeepSORT
tracker = DeepSort(max_age=100, n_init=3, nn_budget=70, max_cosine_distance=0.2, nms_max_overlap=1.0)

# Đọc luồng video từ camera hoặc file
cap = cv2.VideoCapture(0)  # Camera mặc định

tracking_id = None
start_time = stop_time = None
start = stop = None
RED_COLOD = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ camera.")
        break

    # Thay đổi kích thước khung hình
    frame = cv2.resize(frame, (720, 480))  # Giảm kích thước khung hình

    height, width, _ = frame.shape
    coord_text = f'X:0    Y:0'
    location_text = ""

    width_left = width // 2 - 100
    width_right = width // 2 + 100

    results = model.predict(frame, conf=0.5, iou=0.5, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False)

    detections = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box[:6].tolist()
            if cls == 0:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    holistic_results = holistic.process(rgb_frame)

    draw_hand_skeleton(frame, holistic_results)

    if holistic_results.left_hand_landmarks:
        start = is_fist(holistic_results.left_hand_landmarks)
        stop = is_open_hand(holistic_results.left_hand_landmarks)
        person_id = get_person_id_by_hand(tracks, holistic_results.left_hand_landmarks)
        if person_id is not None and start:
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time >= 2:
                tracking_id = person_id
                stop_time = None
        elif person_id is not None and stop:
            if stop_time is None:
                stop_time = time.time()
            elif time.time() - stop_time >= 2:
                tracking_id = None
                start_time = None

    if holistic_results.right_hand_landmarks:
        start = is_fist(holistic_results.right_hand_landmarks)
        stop = is_open_hand(holistic_results.right_hand_landmarks)
        person_id = get_person_id_by_hand(tracks, holistic_results.right_hand_landmarks)
        if person_id is not None and start:
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time >= 2:
                tracking_id = person_id
                stop_time = None
        elif person_id is not None and stop:
            if stop_time is None:
                stop_time = time.time()
            elif time.time() - stop_time >= 2:
                tracking_id = None
                start_time = None

    # Kiểm tra nếu không còn đối tượng nào được theo dõi
    if tracking_id is not None:
        tracking_exists = False
        for track in tracks:
            if track.track_id == tracking_id and track.is_confirmed():
                tracking_exists = True
                break
        
        # Nếu không còn đối tượng nào được theo dõi, reset tracking_id
        if not tracking_exists:
            tracking_id = None

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        bbox = track.to_ltrb()
        x1, y1, x2, y2 = map(int, bbox)

        # Tính toán điểm trung tâm của bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        if track_id == tracking_id:
            color = GREEN_COLOR
            cv2.circle(frame, (center_x, center_y), 5, RED_COLOD, -1)
            coord_text = f'X:{center_x}    Y:{center_y}'
            if width_left < center_x < width_right:
                location_text = "Center"
            elif center_x < width_left:
                location_text = "Left"
            elif center_x > width_right:
                location_text = "Right"
            # draw_body_skeleton(frame, holistic_results)
        else:
            color = RED_COLOD

        label = f"person - {track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        

    cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)
    # cv2.line(frame, (0, height // 2), (width, height // 2), (255, 0, 0), 2)
    cv2.line(frame, (width_left, 0), (width_left, height), (255, 150, 0), 2)
    cv2.line(frame, (width_right, 0), (width_right, height), (255, 150, 0), 2)
    cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
    
    cv2.putText(frame, coord_text, (width // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Thêm thông báo về trạng thái tracking
    if tracking_id is not None:
        status = f"Tracking - ID {tracking_id}"
    else:
        status = "Not Tracking"

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE_COLOR, 2)
    cv2.putText(frame, location_text, (10, height), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK_COLOR, 2)

    cv2.imshow('YOLOv5 + DeepSORT + Mediapipe', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()
