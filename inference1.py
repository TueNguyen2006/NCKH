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

def get_memory_usage():
    process = psutil.Process(os.getpid())  
    mem_info = process.memory_info()      
    ram_usage = mem_info.rss / (1024 ** 2)
    return ram_usage

def distance(p1, p2):
    ''' Calculate distance between two points '''
    p1 = np.array(p1)  
    p2 = np.array(p2)  
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(landmarks, eye):
    ''' Calculate the ratio of the eye length to eye width. '''
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_feature(landmarks):
    ''' Calculate the eye feature as average of EAR for two eyes '''
    return (eye_aspect_ratio(landmarks, left_eye) + \
            eye_aspect_ratio(landmarks, right_eye)) / 2

def mouth_feature(landmarks):
    ''' Calculate mouth feature ratio '''
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def pupil_circularity(landmarks, eye):
    ''' Calculate pupil circularity feature '''
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
                distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
                distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
                distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
                distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
                distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
                distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
                distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4 * math.pi * area) / (perimeter ** 2)

def pupil_feature(landmarks):
    ''' Calculate average pupil circularity for two eyes '''
    return (pupil_circularity(landmarks, left_eye) + \
            pupil_circularity(landmarks, right_eye)) / 2

def normalize_test(poses_array):
    normalized_array = poses_array.copy()
    for dim_idx in [0, 1]:  
        for feature_idx in range(dim_idx, 14, 2):
            normalized_array[feature_idx] = poses_array[feature_idx] - poses_array[dim_idx]  
        
        diff = poses_array[12 + dim_idx] - poses_array[4 + dim_idx]  
        for feature_idx in range(dim_idx, 14, 2):
            normalized_array[feature_idx] = normalized_array[feature_idx] / diff
    return [normalized_array]

def head_pose(face_features):
    face_features_normalized = normalize_test(face_features)
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

def run_face_mp(image, height, width, draw_face=True):
    global alert, running, running_inference, detect
    NOSE, FOREHEAD, LEFT_EYE_IDX, MOUTH_LEFT, CHIN, RIGHT_EYE_IDX, MOUTH_RIGHT = 1, 10, 33, 61, 199, 263, 291
    face_features = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks_positions = []
        for idx, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y]) 
            if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE_IDX, RIGHT_EYE_IDX]:
                face_features.append(data_point.x)
                face_features.append(data_point.y)
        
        landmarks_positions = np.array(landmarks_positions)
        pitch_pred, yaw_pred, roll_pred = head_pose(face_features)
        landmarks_positions[:, 0] *= width
        landmarks_positions[:, 1] *= height

        if draw_face:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
        
        Nose_x, Nose_y = int(landmarks_positions[NOSE][0]), int(landmarks_positions[NOSE][1])
        image = draw_axes(image, pitch_pred, yaw_pred, roll_pred, Nose_x, Nose_y)
        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar / ear
        detect = True
    else:
        ear = mar = puc = moe = -1000
        pitch_pred = yaw_pred = roll_pred = 0
        detect = False
    return ear, mar, puc, moe, pitch_pred, yaw_pred, roll_pred, image

def calibrate(calib_frame_count=200, frames_start=0):
    ears, mars, pucs, moes, pitch_preds, yaw_preds, roll_preds = [], [], [], [], [], [], []
    cap = cv2.VideoCapture(0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = 0
    while True:
        success, image = cap.read()
        if not success: break
        frames += 1
        ear, mar, puc, moe, pitch_p, yaw_p, roll_p, image = run_face_mp(image, height=height, width=width)
        if ear != -1000 and frames > frames_start:
            ears.append(ear); mars.append(mar); pucs.append(puc); moes.append(moe)
            pitch_preds.append(pitch_p); yaw_preds.append(yaw_p); roll_preds.append(roll_p)

        cv2.putText(image, "Calibration", (int(0.02*image.shape[1]), int(0.14*image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        cv2.imshow('MediaPipe FaceMesh', image)
        if (cv2.waitKey(5) & 0xFF == ord("q")) or frames >= frames_start + calib_frame_count: break
    
    cv2.destroyAllWindows(); cap.release()
    return [np.mean(ears), np.std(ears)], [np.mean(mars), np.std(mars)], [np.mean(pucs), np.std(pucs)], \
           [np.mean(moes), np.std(moes)], np.mean(pitch_preds), np.mean(yaw_preds), np.mean(roll_preds)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plot_counter = 0

def get_classification(input_data, save_plots=False):
    global plot_counter
    model_input = torch.FloatTensor(np.array([input_data[i:i+5] for i in range(0, 16, 3)]))
    with torch.no_grad():
        preds = model(model_input)
        preds = (preds > 0.5).int().cpu().numpy()
    label = int(preds.sum() >= 5)

    if save_plots:
        plot_counter += 1
        folder_name = f"{plot_counter}_{label}"
        os.makedirs(folder_name, exist_ok=True)
        input_array = np.array(input_data)
        feature_names = ['EAR', 'MAR', 'PUC', 'MoE']
        for i, feat_name in enumerate(feature_names):
            plt.figure(figsize=(6, 4))
            plt.plot(input_array[:, i], linewidth=3)
            plt.xlabel('Frames'); plt.ylabel('Value')
            plt.legend([feat_name])
            plt.savefig(os.path.join(folder_name, f"{feat_name}_plot.png"))
            plt.close('all')
    return label

def infer(ears_norm, mars_norm, pucs_norm, moes_norm, pitch_pred_norm, yaw_pred_norm, roll_pred_norm, count_detect_drownsiness=6):
    global running, running_inference, alert, detect
    ear_main = mar_main = puc_main = moe_main = pitch_main = yaw_main = roll_main = head = head_count = 0
    decay, frame_before_run, count_decision, processed_frames = 0.9, 0, 0, 0
    input_data = []
    label = None
    
    cap = cv2.VideoCapture(0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened() and running:
        success, image = cap.read()
        if not success: break
        processed_frames += 1
        ear, mar, puc, moe, pitch_p, yaw_p, roll_p, image = run_face_mp(image, height=height, width=width)

        if running_inference:
            if ear != -1000:
                ear = (ear - ears_norm[0])/ears_norm[1]
                mar = (mar - mars_norm[0])/mars_norm[1]
                puc = (puc - pucs_norm[0])/pucs_norm[1]
                moe = (moe - moes_norm[0])/moes_norm[1]
                pitch_main = pitch_p - pitch_pred_norm
                if ear_main == -1000:
                    ear_main, mar_main, puc_main, moe_main = ear, mar, puc, moe
                else:
                    ear_main = ear_main*decay + (1-decay)*ear
                    mar_main = mar_main*decay + (1-decay)*mar
                    puc_main = puc_main*decay + (1-decay)*puc
                    moe_main = moe_main*decay + (1-decay)*moe
            else:
                ear_main = mar_main = puc_main = moe_main = -1000

            if detect:
                if pitch_main > 0.3 or pitch_main < -0.2:
                    head, head_count = 1, head_count + 1
                else:
                    head, head_count = 0, 0
            
            if len(input_data) == 20: input_data.pop(0)
            input_data.append([ear_main, mar_main, puc_main, moe_main])
            frame_before_run += 1

            if frame_before_run >= 15 and len(input_data) == 20:
                frame_before_run = 0
                label = get_classification(input_data)
                count_decision = 0 if label == 0 else count_decision + 1
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
            
            if count_decision >= count_detect_drownsiness or (head == 1 and head_count >= 20):
                print("Drowsiness detected!")
        else:
            image.fill(0); count_decision = head_count = 0

        cv2.imshow('MediaPipe FaceMesh', image)
        out.write(image)
        if processed_frames % 100 == 0: gc.collect(); processed_frames = 0
        if cv2.waitKey(5) & 0xFF == ord("q"): running = False

    cv2.destroyAllWindows(); cap.release(); out.release()

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=UserWarning)
    right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
    left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
    mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]
    states = ['normal', 'drowsy']
    running = running_inference = True
    
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.8, max_num_faces=1)
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    model_head_pose = pickle.load(open('./models/model.pkl', 'rb'))
    model = torch.jit.load('./models/clf_lstm.pth')
    model.eval()

    print('Starting calibration...')
    ears_n, mars_n, pucs_n, moes_n, p_p, y_p, r_p = calibrate()
    print('Starting main application')
    infer(ears_n, mars_n, pucs_n, moes_n, p_p, y_p, r_p)
