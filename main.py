import cv2 
import mediapipe as mp
import math
import numpy as np 
import os 
import torch
import psutil
import pickle
import warnings
import gc
from ras import Ras



def get_memory_usage():
    process = psutil.Process(os.getpid())  
    mem_info = process.memory_info()       
    ram_usage = mem_info.rss / (1024 ** 2) 
    return ram_usage

def distance(p1, p2):
    ''' Calculate distance between two points
    :param p1: First Point 
    :param p2: Second Point
    :return: Euclidean distance between the points. (Using only the x and y coordinates).
    '''
    p1 = np.array(p1)  # Chuyển đổi nếu p1 là list hoặc tuple
    p2 = np.array(p2)  # Chuyển đổi nếu p2 là list hoặc tuple
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(landmarks, eye):
    ''' Calculate the ratio of the eye length to eye width. 
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Eye aspect ratio value
    '''
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_feature(landmarks):
    ''' Calculate the eye feature as the average of the eye aspect ratio for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Eye feature value
    '''
    return (eye_aspect_ratio(landmarks, left_eye) + \
    eye_aspect_ratio(landmarks, right_eye))/2

def mouth_feature(landmarks):
    ''' Calculate mouth feature as the ratio of the mouth length to mouth width
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Mouth feature value
    '''
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3)/(3*D)

def pupil_circularity(landmarks, eye):
    ''' Calculate pupil circularity feature.
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Pupil circularity for the eye coordinates
    '''
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
            distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
            distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
            distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
            distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
            distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
            distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
            distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4*math.pi*area)/(perimeter**2)

def pupil_feature(landmarks):
    ''' Calculate the pupil feature as the average of the pupil circularity for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Pupil feature value
    '''
    return (pupil_circularity(landmarks, left_eye) + \
        pupil_circularity(landmarks, right_eye))/2

def normalize_test(poses_array):
    # poses_array: array với các giá trị theo thứ tự ['nose_x', 'nose_y', ..., 'mouth_right_y']
    normalized_array = poses_array.copy()
    # Indices của các feature trong array
    
    for dim_idx in [0, 1]:  # X và Y tương ứng là chỉ số chẵn/lẻ
        # Centering around the nose
        for feature_idx in range(dim_idx, 14, 2):
            normalized_array[feature_idx] = poses_array[feature_idx] - poses_array[dim_idx]  # Trừ giá trị nose_x hoặc nose_y
        
        # Scaling
        diff = poses_array[12 + dim_idx] - poses_array[4 + dim_idx]  # mouth_right_dim - left_eye_dim
        for feature_idx in range(dim_idx, 14, 2):
            normalized_array[feature_idx] = normalized_array[feature_idx] / diff

    # Tạo array 2D để phù hợp với đầu vào của mô hình
    face_features_array = [normalized_array]
    return face_features_array


def head_pose(face_features):
    
    pitch_pred, yaw_pred, roll_pred = 0, 0, 0

    face_features_normalized = normalize_test(face_features)
        
    # Dự đoán các giá trị pitch, yaw, roll
    pitch_pred, yaw_pred, roll_pred = model_head_pose.predict(face_features_normalized).ravel()

    return pitch_pred, yaw_pred, roll_pred

def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
    axes_points = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(int)
    axes_points[0, :] = axes_points[0, :] + tx
    axes_points[1, :] = axes_points[1, :] + ty
    
    new_img = img.copy()
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)    
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    return new_img

def run_face_mp(image, height, width, draw_face = True):
    global alert, running, running_inference, arduino, servo_delta_x, servo_delta_y, current_servo_x, current_servo_y, detect
    NOSE = 1
    FOREHEAD = 10
    LEFT_EYE = 33
    MOUTH_LEFT = 61
    CHIN = 199
    RIGHT_EYE = 263
    MOUTH_RIGHT = 291
    face_features = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks_positions = []
        # assume that only face is present in the image
        for idx, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y]) # saving normalized landmark positions
            if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]:
                face_features.append(data_point.x)
                face_features.append(data_point.y)
        
        landmarks_positions = np.array(landmarks_positions)
        pitch_pred, yaw_pred, roll_pred = head_pose(face_features)

        landmarks_positions[:, 0] *= width
        landmarks_positions[:, 1] *= height

        # draw face mesh over image
        if draw_face:
            for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
        
        Nose_x = int(landmarks_positions[NOSE][0])
        Nose_y = int(landmarks_positions[NOSE][1])

        image = draw_axes(image, pitch_pred, yaw_pred, roll_pred, Nose_x, Nose_y)

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar/ear
        detect = True
    else:
        ear = -1000
        mar = -1000
        puc = -1000
        moe = -1000
        pitch_pred, yaw_pred, roll_pred = 0, 0, 0
        detect = False
    results = None
    return ear, mar, puc, moe, pitch_pred, yaw_pred, roll_pred, image


def calibrate(calib_frame_count=200, frames_start = 150):

    ears = []
    mars = []
    pucs = []
    moes = []
    pitch_preds = []
    yaw_preds = []
    roll_preds = []

    cap = cv2.VideoCapture(0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # cap.set(3, width)
    # cap.set(4, height)
    frames = 0
    
    while True:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        frames +=1
        ear, mar, puc, moe, pitch_pred, yaw_pred, roll_pred, image = run_face_mp(image, height=height, width=width)
        if ear != -1000 and frames > frames_start:
            ears.append(ear)
            mars.append(mar)
            pucs.append(puc)
            moes.append(moe)
            pitch_preds.append(pitch_pred)
            yaw_preds.append(yaw_pred)
            roll_preds.append(roll_pred)

        cv2.putText(image, "Calibration", (int(0.02*image.shape[1]), int(0.14*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        if frames >= frames_start + calib_frame_count:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    ears = np.array(ears)
    mars = np.array(mars)
    pucs = np.array(pucs)
    moes = np.array(moes)

    pitch_preds = np.array(pitch_preds)
    pitch_mean_zscore = pitch_preds.mean()  # Lấy giá trị trung bình sau khi chuẩn hóa

    yaw_preds = np.array(yaw_preds)
    yaw_mean_zscore = yaw_preds.mean()  # Lấy giá trị trung bình sau khi chuẩn hóa

    roll_preds = np.array(roll_preds)
    roll_mean_zscore = roll_preds.mean()  # Lấy giá trị trung bình sau khi chuẩn hóa

    return [ears.mean(), ears.std()], [mars.mean(), mars.std()], \
        [pucs.mean(), pucs.std()], [moes.mean(), moes.std()], \
        pitch_mean_zscore, yaw_mean_zscore, roll_mean_zscore


def get_classification(input_data):
    ''' Perform classification over the facial  features.
    :param input_data: List of facial features for 20 frames
    :return: Alert / Drowsy state prediction
    '''
    model_input = []
    model_input.append(input_data[0:5])
    model_input.append(input_data[3:8])
    model_input.append(input_data[6:11])
    model_input.append(input_data[9:14])
    model_input.append(input_data[12:17])
    model_input.append(input_data[15:20])
    
    model_input = [input_data[i:i+5] for i in range(0, 16, 3)]
    model_input = torch.FloatTensor(np.array(model_input))
    with torch.no_grad():
        preds = model(model_input)
        preds = (preds > 0.5).int().cpu().numpy()
    return int(preds.sum() >= 5)

# def load_models():
#     cnn = torch.jit.load("cnn_model_jit.pt")
#     cnn.eval()
#     svm = joblib.load("svm_model.pkl")
#     scaler = joblib.load("scaler.pkl")
#     return cnn, svm, scaler

# def get_classification(input_data):
#     input_tensor = torch.tensor(input_data, dtype=torch.float32).T.unsqueeze(0).unsqueeze(0)
#     with torch.no_grad():
#         features = cnn(input_tensor).numpy()

#     # Chuẩn hóa dữ liệu
#     features = scaler.transform(features)

#     # Dự đoán với SVM
#     prediction = svm.predict(features)
#     return prediction[0]

# def get_classification(input_data):
#     model_inputs = [
#         input_data[0:20],
#         input_data[5:25],
#         input_data[10:30],
#         input_data[15:35],
#         input_data[20:40],
#         input_data[25:45]
#     ]
    
#     predictions = []
    
#     for model_input in model_inputs:
#         input_tensor = torch.tensor(model_input).unsqueeze(0).unsqueeze(0).float()
        
#         with torch.no_grad():
#             features = cnn(input_tensor).numpy()
        
#         # Chuẩn hóa dữ liệu
#         features = scaler.transform(features)
        
#         # Dự đoán với SVM
#         prediction = svm.predict(features)
#         predictions.append(prediction[0])
    
#     # Majority voting: nếu tổng số dự đoán buồn ngủ >= 5, trả về 1 (buồn ngủ), ngược lại 0
#     return 1 if sum(predictions) > 5 else 0


def infer(ears_norm, mars_norm, pucs_norm, moes_norm, pitch_pred_norm, yaw_pred_norm, roll_pred_norm, count_detect_drownsiness = 6, ras_obj=None):
    ''' Perform inference.
    :param ears_norm: Normalization values for eye feature
    :param mars_norm: Normalization values for mouth feature
    :param pucs_norm: Normalization values for pupil feature
    :param moes_norm: Normalization values for mouth over eye feature. 
    :param 
    '''
    global running, running_inference, alert, detect

    ear_main = 0
    mar_main = 0
    puc_main = 0
    moe_main = 0
    pitch_main = 0
    yaw_main = 0
    roll_main = 0
    head = 0
    head_count = 0
    decay = 0.9 # use decay to smoothen the noise in feature values

    label = None
    processed_frames = 0

    input_data = []
    frame_before_run = 0
    count_decision = 0 
    cap = cv2.VideoCapture(0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # cap.set(3, width)
    # cap.set(4, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    while cap.isOpened() and running:

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        processed_frames += 1
        ear, mar, puc, moe, pitch_pred, yaw_pred, roll_pred, image = run_face_mp(image, height=height, width=width)

        if running_inference: # Nếu xe đang chạy, chạy inference và ngược lại
            
            if ear != -1000:
                ear = (ear - ears_norm[0])/ears_norm[1]
                mar = (mar - mars_norm[0])/mars_norm[1]
                puc = (puc - pucs_norm[0])/pucs_norm[1]
                moe = (moe - moes_norm[0])/moes_norm[1]
                pitch_main = pitch_pred - pitch_pred_norm
                # yaw_main = yaw_pred - yaw_pred_norm
                # roll_main = roll_pred - roll_pred_norm
                if ear_main == -1000:
                    ear_main = ear
                    mar_main = mar
                    puc_main = puc
                    moe_main = moe
                else:
                    ear_main = ear_main*decay + (1-decay)*ear #EMA data smoothing
                    mar_main = mar_main*decay + (1-decay)*mar
                    puc_main = puc_main*decay + (1-decay)*puc
                    moe_main = moe_main*decay + (1-decay)*moe
            else:
                ear_main = -1000
                mar_main = -1000    
                puc_main = -1000
                moe_main = -1000
                pitch_main = pitch_main
                # yaw_main = 0
                # roll_main = 0
            if detect:
                if pitch_main > 0.3 or pitch_main < - 0.2:
                    head = 1
                    head_count += 1
                else:
                    head = 0
                    head_count = 0
            else:
                if pitch_main > 0.2 or pitch_main < -0.15:
                    head_count += 1
                    head = 1
                else:
                    head_count = 0
                
            if len(input_data) == 20:
                input_data.pop(0)
            input_data.append([ear_main, mar_main, puc_main, moe_main])

            frame_before_run += 1

            if frame_before_run >= 15 and len(input_data) == 20:
                frame_before_run = 0
                label = get_classification(input_data) # 1 is drowsiness, 0 is normal
                # print ('got label ', label)
                if label == 0:
                    count_decision = 0 # Reset count_decision if predict no drownsiness
                else:
                    count_decision += 1
                print(f"Count decision: {count_decision}")

            cv2.putText(image, "EAR: %.2f" %(ear_main), (int(0.02*width), int(0.07*height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(image, "MAR: %.2f" %(mar_main), (int(0.27*width), int(0.07*height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(image, "PUC: %.2f" %(puc_main), (int(0.52*width), int(0.07*height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(image, "MOE: %.2f" %(moe_main), (int(0.77*width), int(0.07*height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Prepare text to display on the screen
            angle_text_pitch = f"Pitch: {pitch_main:.2f}°"
            angle_text_yaw = f"Yaw: {yaw_main:.2f}°"
            angle_text_roll = f"Roll: {roll_main:.2f}°"

            # Display the angle values on the image
            cv2.putText(image, angle_text_pitch, (int(0.02*width), int(0.1*height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, angle_text_yaw, (int(0.02*width), int(0.2*height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, angle_text_roll, (int(0.02*width), int(0.3*height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            if label is not None:
                if label == 0:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                cv2.putText(image, "%s" %(states[label]), (int(0.4*image.shape[1]), int(0.22*image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            if count_decision >= count_detect_drownsiness or (head == 1 and head_count >=20): # Turn Alert
                print("Drowsiness detected! Activating alert.")
                ras_obj.warning()
            else:
                ras_obj.turn_off()
                pass
        else:
            image.fill(0)
            count_decision = 0
            head_count = 0
            alert = False
        cv2.imshow('MediaPipe FaceMesh', image)
        # out.write(image)

        if processed_frames % 100 == 0:
            gc.collect()
            processed_frames = 0

        if cv2.waitKey(5) & 0xFF == ord("q"):
            running = False
    running = False
    alert = False
    cv2.destroyAllWindows()
    cap.release()
    # out.release()


if __name__ == "__main__":
    try:
        warnings.filterwarnings("ignore", category=UserWarning)

        right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
        left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
        mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]
        states = ['normal', 'drowsy']

        ras_obj = Ras(serial_port="/dev/serial0", baudrate=9600, vibration_motor=17)

        running = True  
        running_inference = True
        alert = False
        stop_thread = False

        servo_delta_x = 0
        servo_delta_y = 0
        current_servo_x = 90
        current_servo_y = 90


        # Khởi tạo FaceMesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.8,
            max_num_faces=1)
        mp_drawing = mp.solutions.drawing_utils 
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        model_head_pose = pickle.load(open('./models/model.pkl', 'rb'))
        model_lstm_path = './models/clf_lstm.pth'
        model = torch.jit.load(model_lstm_path)
        model.eval()

        print('Starting calibration. Please be in neutral state')
        ears_norm, mars_norm, pucs_norm, moes_norm, pitch_pred, yaw_pred, roll_pred = calibrate()
        print(ears_norm, mars_norm, pucs_norm, moes_norm, pitch_pred, yaw_pred, roll_pred)
        print('Starting main application')
        infer(ears_norm, mars_norm, pucs_norm, moes_norm, pitch_pred, yaw_pred, roll_pred)
    
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    
    except Exception as e:
        print("An error occurred:", e)
    
    finally:
        # Gọi dọn dẹp tài nguyên:
        print("Cleaning up hardware resources...")
        ras_obj.close()
        cv2.destroyAllWindows()
        print("Program terminated.")