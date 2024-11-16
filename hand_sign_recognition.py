import cv2
from gesture_recognition import GestureRecognition  # Đảm bảo import đúng
from cvfpscalc import CvFpsCalc

class HandSignRecognition:

    def __init__(self):
        # Tạo đối tượng nhận diện cử chỉ và load mô hình
        self.gesture_detector = GestureRecognition("C:/Users/nhh17/OneDrive/Documents/ros_hand_gesture_recognition-main/src/model/keypoint_classifier/keypoint_classifier_label.csv",
                                                   "C:/Users/nhh17/OneDrive/Documents/ros_hand_gesture_recognition-main/src/model/keypoint_classifier/keypoint_classifier.tflite")
        self.cv_fps_calc = CvFpsCalc(buffer_len=10)
        self.cap = cv2.VideoCapture(1)  # Sử dụng camera mặc định

    def run(self):
        while True:
            ret, frame = self.cap.read()  # Đọc hình ảnh từ camera
            if not ret:
                break
            
            debug_image, gesture = self.gesture_detector.recognize(frame)
            print("Detected gesture:", gesture)  # Xuất cử chỉ nhận diện

            # Vẽ thông tin FPS lên hình ảnh
            fps = self.cv_fps_calc.get()
            debug_image = self.gesture_detector.draw_fps_info(debug_image, fps)

            cv2.imshow('Gesture Recognition', debug_image)  # Hiển thị hình ảnh

            # Dừng khi nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()  # Giải phóng camera
        cv2.destroyAllWindows()  # Đóng tất cả cửa sổ

if __name__ == "__main__":
    hand_sign = HandSignRecognition()
    hand_sign.run()
