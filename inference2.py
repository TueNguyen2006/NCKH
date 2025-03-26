import cv2
import mediapipe as mp
import math
import numpy as np
import time
import torch
import threading
import serial
import pickle
import warnings
import RPi.GPIO as GPIO
import logging
import gc

# ---------------------- Setup Logging ----------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------- Hardware Control ----------------------
class Ras:
    def __init__(self, serial_port="/dev/serial0", baudrate=9600, vibration_motor=17):
        """Initialize Raspberry Pi hardware for audio and vibration alert."""
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.ser = None
        self.serial_ready = threading.Event()
        self.vibration_motor = vibration_motor
        self.running_event = threading.Event()  # Controls the alert (audio + vibration)

        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.vibration_motor, GPIO.OUT)

        # Start serial connection in a separate thread
        self.serial_thread = threading.Thread(target=self._serial_connect, daemon=True)
        self.serial_thread.start()

        # Start threads for music and motor
        self.music_thread = threading.Thread(target=self._play_music, daemon=True)
        self.motor_thread = threading.Thread(target=self._vibrate_motor, daemon=True)
        self.music_thread.start()
        self.motor_thread.start()

    def _serial_connect(self):
        """Continuously try to connect to Serial until successful."""
        while True:
            try:
                self.ser = serial.Serial(port=self.serial_port, baudrate=self.baudrate, timeout=1)
                logging.info("Serial connected successfully!")
                self.serial_ready.set()
                break
            except serial.SerialException as e:
                logging.error("Serial connection error: %s", e)
                time.sleep(1)

    def _get_serial(self):
        """Wait until serial is ready and return it."""
        self.serial_ready.wait()
        return self.ser

    def _send_command(self, command):
        """Send command bytes via serial."""
        ser = self._get_serial()
        ser.write(command)

    def play_song(self, song_number):
        """Send command to play song with given song number."""
        command = bytes([0x7E, 0xFF, 0x06, 0x03, 0x00, 0x00, song_number, 0xEF])
        self._send_command(command)

    def set_volume(self, volume_level):
        """Send command to set volume (0-30)."""
        command = bytes([0x7E, 0xFF, 0x06, 0x06, 0x00, 0x00, volume_level, 0xEF])
        self._send_command(command)

    def _play_music(self, volume_level=30, song_number=1):
        """Continuously play music when alert is activated."""
        while True:
            self.running_event.wait()
            logging.info("Playing alert music...")
            self.set_volume(volume_level)
            self.play_song(song_number)
            time.sleep(2)  # Delay between songs

    def _vibrate_motor(self, duration=1):
        """Continuously vibrate motor when alert is activated."""
        while True:
            self.running_event.wait()
            logging.info("Activating vibration motor...")
            GPIO.output(self.vibration_motor, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(self.vibration_motor, GPIO.LOW)
            time.sleep(0.5)

    def warning(self):
        """Activate alert if not already running."""
        if self.running_event.is_set():
            logging.info("Alert already active.")
            return
        logging.warning("Activating alert!")
        self.running_event.set()

    def turn_off(self):
        """Deactivate alert if active."""
        if not self.running_event.is_set():
            logging.info("Alert is not active; nothing to turn off.")
            return
        logging.info("Deactivating alert...")
        self.running_event.clear()

    def close(self):
        """Clean up hardware resources."""
        self.turn_off()
        if self.ser:
            self.ser.close()
        GPIO.cleanup()
        logging.info("Hardware resources cleaned up.")

# ---------------------- Feature Extraction ----------------------
class FeatureExtractor:
    def __init__(self, left_eye, right_eye, mouth):
        """Initialize with landmark indices for left_eye, right_eye, and mouth."""
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.mouth = mouth

    @staticmethod
    def distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def eye_aspect_ratio(self, landmarks, eye):
        N1 = self.distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
        N2 = self.distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
        N3 = self.distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
        D = self.distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
        return (N1 + N2 + N3) / (3 * D)

    def eye_feature(self, landmarks):
        return (self.eye_aspect_ratio(landmarks, self.left_eye) +
                self.eye_aspect_ratio(landmarks, self.right_eye)) / 2

    def mouth_feature(self, landmarks):
        N1 = self.distance(landmarks[self.mouth[1][0]], landmarks[self.mouth[1][1]])
        N2 = self.distance(landmarks[self.mouth[2][0]], landmarks[self.mouth[2][1]])
        N3 = self.distance(landmarks[self.mouth[3][0]], landmarks[self.mouth[3][1]])
        D = self.distance(landmarks[self.mouth[0][0]], landmarks[self.mouth[0][1]])
        return (N1 + N2 + N3) / (3 * D)

    def pupil_circularity(self, landmarks, eye):
        perimeter = (
            self.distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) +
            self.distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) +
            self.distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) +
            self.distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) +
            self.distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) +
            self.distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) +
            self.distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) +
            self.distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
        )
        area = math.pi * ((self.distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
        return (4 * math.pi * area) / (perimeter ** 2)

    def pupil_feature(self, landmarks):
        return (self.pupil_circularity(landmarks, self.left_eye) +
                self.pupil_circularity(landmarks, self.right_eye)) / 2

# ---------------------- Drowsiness Detector ----------------------
class DrowsinessDetector:
    def __init__(self, face_mesh, model_head_pose, model_lstm, feature_extractor, calibration_params=None):
        """
        :param face_mesh: Instance of MediaPipe FaceMesh.
        :param model_head_pose: Pre-trained head pose estimation model (loaded via pickle).
        :param model_lstm: Pre-trained LSTM model (loaded via torch.jit.load).
        :param feature_extractor: Instance of FeatureExtractor.
        :param calibration_params: Optional tuple with calibration values.
        """
        self.face_mesh = face_mesh
        self.model_head_pose = model_head_pose
        self.model_lstm = model_lstm
        self.feature_extractor = feature_extractor
        self.calibration_params = calibration_params
        self.input_data = []
        self.frame_skip = 2  # Process every 2nd frame (có thể điều chỉnh)
        self.skip_counter = 0

    def _normalize_features(self, feature_value, mean_std):
        mean, std = mean_std
        return (feature_value - mean) / std

    def run_face_mp(self, image, width, height, draw_face=True):
        """
        Process one frame với MediaPipe FaceMesh, tính toán các đặc trưng khuôn mặt và head pose.
        Trả về tuple: (ear, mar, puc, moe, pitch, yaw, roll, image_processed)
        """
        # Các chỉ số điểm cần lấy (có thể tùy chỉnh nếu cần)
        NOSE = 1; FOREHEAD = 10; LEFT_EYE = 33; MOUTH_LEFT = 61
        CHIN = 199; RIGHT_EYE = 263; MOUTH_RIGHT = 291
        face_features = []

        # Chuyển đổi màu và disable writeable để tối ưu hiệu năng
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            landmarks = []
            # Giả sử chỉ có một khuôn mặt
            for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
                landmarks.append([lm.x, lm.y])
                # Lấy các điểm đặc trưng để tính head pose
                if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]:
                    face_features.extend([lm.x, lm.y])
            landmarks = np.array(landmarks)
            # Head pose: sử dụng mô hình đã load từ pickle
            pitch, yaw, roll = self.model_head_pose.predict([face_features]).ravel()

            # Nhân tọa độ lên theo kích thước ảnh
            landmarks[:, 0] *= width
            landmarks[:, 1] *= height

            # Vẽ lưới khuôn mặt nếu cần
            if draw_face:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

            # Vẽ trục tọa độ lên mũi
            Nose_x, Nose_y = int(landmarks[NOSE][0]), int(landmarks[NOSE][1])
            image = draw_axes(image, pitch, yaw, roll, Nose_x, Nose_y)

            # Tính các đặc trưng
            ear = self.feature_extractor.eye_feature(landmarks)
            mar = self.feature_extractor.mouth_feature(landmarks)
            puc = self.feature_extractor.pupil_feature(landmarks)
            moe = mar / ear
            detected = True
        else:
            ear = mar = puc = moe = -1000
            pitch = yaw = roll = 0
            detected = False

        return ear, mar, puc, moe, pitch, yaw, roll, image, detected

    def calibrate(self, cap, calib_frame_count=200, frames_start=150):
        """
        Hiệu chỉnh hệ thống bằng cách thu thập các đặc trưng khi người lái ở trạng thái trung tính.
        Trả về tuple các giá trị (mean, std) cho EAR, MAR, PUC, MOE và trung bình head pose (pitch, yaw, roll).
        """
        ears, mars, pucs, moes = [], [], [], []
        pitch_vals, yaw_vals, roll_vals = [], [], []
        frames = 0

        logging.info("Starting calibration. Please maintain a neutral state.")
        while True:
            success, frame = cap.read()
            if not success:
                logging.warning("Empty camera frame during calibration.")
                continue
            frames += 1

            # Có thể giảm độ phân giải để tăng FPS (ví dụ scale 0.5)
            frame = cv2.resize(frame, None, fx=0.75, fy=0.75)

            width, height = frame.shape[1], frame.shape[0]
            ear, mar, puc, moe, pitch, yaw, roll, proc_frame, _ = self.run_face_mp(frame, width, height)
            if ear != -1000 and frames > frames_start:
                ears.append(ear)
                mars.append(mar)
                pucs.append(puc)
                moes.append(moe)
                pitch_vals.append(pitch)
                yaw_vals.append(yaw)
                roll_vals.append(roll)

            cv2.putText(proc_frame, "Calibration", (int(0.02*width), int(0.14*height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            cv2.imshow('Calibration', proc_frame)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
            if frames >= frames_start + calib_frame_count:
                break

        cv2.destroyWindow('Calibration')
        ears = np.array(ears)
        mars = np.array(mars)
        pucs = np.array(pucs)
        moes = np.array(moes)
        pitch_mean = np.mean(pitch_vals)
        yaw_mean = np.mean(yaw_vals)
        roll_mean = np.mean(roll_vals)
        calibration = {
            'ears': (ears.mean(), ears.std()),
            'mars': (mars.mean(), mars.std()),
            'pucs': (pucs.mean(), pucs.std()),
            'moes': (moes.mean(), moes.std()),
            'head_pose': (pitch_mean, yaw_mean, roll_mean)
        }
        logging.info("Calibration complete: %s", calibration)
        return calibration

    def get_classification(self, input_data):
        """Chia nhỏ chuỗi dữ liệu và chạy mô hình LSTM để dự đoán trạng thái buồn ngủ."""
        # Ví dụ: tách input_data thành các đoạn con, tạo tensor và dự đoán
        model_input = [input_data[i:i+5] for i in range(0, 16, 3)]
        model_input = torch.FloatTensor(np.array(model_input))
        with torch.no_grad():
            preds = self.model_lstm(model_input)
            preds = (preds > 0.5).int().cpu().numpy()
        # Nếu số lượng dự đoán buồn ngủ >= 5/6, trả về 1 (buồn ngủ)
        return int(preds.sum() >= 5)

    def run_inference(self, ras_obj, count_detect_drowsiness=6):
        """
        Chạy vòng lặp nhận diện, dự đoán và kích hoạt cảnh báo nếu cần.
        ras_obj: đối tượng của lớp Ras để kích hoạt cảnh báo.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Cannot open camera.")
            return

        # Thiết lập độ phân giải thấp hơn (ví dụ 480x360)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
        
        count_decision = 0
        frame_before_run = 0
        head_count = 0
        decay = 0.9
        ear_main = mar_main = puc_main = moe_main = 0
        pitch_main = 0

        running = True
        processed_frames = 0

        while cap.isOpened() and running:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Empty frame received.")
                continue

            # Áp dụng frame skipping
            self.skip_counter += 1
            if self.skip_counter < self.frame_skip:
                continue
            else:
                self.skip_counter = 0

            processed_frames += 1
            ear, mar, puc, moe, pitch, yaw, roll, proc_frame, detected = self.run_face_mp(frame, width, height)
            if detected and ear != -1000:
                # Sử dụng các giá trị hiệu chuẩn từ calibration
                cal = self.calibration_params
                ear = self._normalize_features(ear, cal['ears'])
                mar = self._normalize_features(mar, cal['mars'])
                puc = self._normalize_features(puc, cal['pucs'])
                moe = self._normalize_features(moe, cal['moes'])
                pitch_main = pitch - cal['head_pose'][0]
                
                # Data smoothing: exponential moving average
                ear_main = ear if ear_main == 0 else ear_main * decay + (1-decay) * ear
                mar_main = mar if mar_main == 0 else mar_main * decay + (1-decay) * mar
                puc_main = puc if puc_main == 0 else puc_main * decay + (1-decay) * puc
                moe_main = moe if moe_main == 0 else moe_main * decay + (1-decay) * moe
            else:
                ear_main = mar_main = puc_main = moe_main = -1000

            frame_before_run += 1
            if frame_before_run >= 15:
                frame_before_run = 0
                if len(self.input_data) == 20:
                    self.input_data.pop(0)
                self.input_data.append([ear_main, mar_main, puc_main, moe_main])
                if len(self.input_data) == 20:
                    label = self.get_classification(self.input_data)
                    if label == 0:
                        count_decision = 0
                    else:
                        count_decision += 1
                    logging.debug("Count decision: %s", count_decision)
            # Hiển thị thông tin lên ảnh
            cv2.putText(proc_frame, f"EAR: {ear_main:.2f}", (int(0.02*width), int(0.07*height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            cv2.putText(proc_frame, f"MAR: {mar_main:.2f}", (int(0.27*width), int(0.07*height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            cv2.putText(proc_frame, f"PUC: {puc_main:.2f}", (int(0.52*width), int(0.07*height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            cv2.putText(proc_frame, f"MOE: {moe_main:.2f}", (int(0.77*width), int(0.07*height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            cv2.putText(proc_frame, f"Pitch: {pitch_main:.2f}°", (int(0.02*width), int(0.1*height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(proc_frame, f"Yaw: {yaw:.2f}°", (int(0.02*width), int(0.2*height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(proc_frame, f"Roll: {roll:.2f}°", (int(0.02*width), int(0.3*height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # Nếu điều kiện buồn ngủ đạt ngưỡng, kích hoạt cảnh báo
            if count_decision >= count_detect_drowsiness or head_count >= 20:
                logging.warning("Drowsiness detected! Activating alert.")
                ras_obj.warning()
            else:
                ras_obj.turn_off()

            cv2.imshow('Drowsiness Detector', proc_frame)
            out.write(proc_frame)

            # Gọi GC sau mỗi 100 frame để thu hồi bộ nhớ không cần thiết
            if processed_frames % 100 == 0:
                gc.collect()

            if cv2.waitKey(5) & 0xFF == ord("q"):
                running = False
                break

        cv2.destroyAllWindows()
        cap.release()
        out.release()

# ---------------------- Utility Functions ----------------------
def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    """Vẽ các trục tọa độ lên ảnh dựa trên giá trị head pose."""
    yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
    axes_points = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]], dtype=np.float64)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(int)
    axes_points[0, :] += tx
    axes_points[1, :] += ty
    new_img = img.copy()
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    return new_img

# ---------------------- Main Function ----------------------
def main():
    try:
        warnings.filterwarnings("ignore", category=UserWarning)
        # Define landmark indices (using consistent English naming)
        left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
        right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
        mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]
        states = ['normal', 'drowsy']  # Có thể dùng cho hiển thị nếu cần

        # Initialize hardware alert system
        ras_obj = Ras(serial_port="/dev/serial0", baudrate=9600, vibration_motor=17)

        # Initialize MediaPipe FaceMesh với max_num_faces=1 để tiết kiệm tài nguyên
        global mp_drawing, drawing_spec, mp_face_mesh  # Sử dụng global để dùng chung trong các hàm vẽ (có thể cải tiến thêm)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3,
                                          min_tracking_confidence=0.8,
                                          max_num_faces=1)
        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # Load models
        model_head_pose = pickle.load(open('./models/model.pkl', 'rb'))
        model_lstm = torch.jit.load('./models/clf_lstm.pth')
        model_lstm.eval()

        # Initialize FeatureExtractor with landmark indices
        feature_extractor = FeatureExtractor(left_eye, right_eye, mouth)
        # Initialize DrowsinessDetector with loaded models and face_mesh
        detector = DrowsinessDetector(face_mesh, model_head_pose, model_lstm, feature_extractor)

        # Open a camera for calibration
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Cannot open camera for calibration.")
            return

        calibration_params = detector.calibrate(cap)
        detector.calibration_params = calibration_params
        cap.release()

        # Start main inference loop
        logging.info("Starting main application...")
        detector.run_inference(ras_obj)

    except KeyboardInterrupt:
        logging.info("Program interrupted by user.")
    except Exception as e:
        logging.error("An error occurred: %s", e)
    finally:
        logging.info("Cleaning up hardware resources...")
        ras_obj.close()
        cv2.destroyAllWindows()
        logging.info("Program terminated.")

if __name__ == "__main__":
    main()
