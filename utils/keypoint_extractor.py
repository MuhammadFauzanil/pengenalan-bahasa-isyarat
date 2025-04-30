import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)

def extract_keypoints(image, return_landmarks=False):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # Ambil keypoints tangan kiri dan kanan
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))

    keypoints = np.concatenate([lh.flatten(), rh.flatten()])

    if return_landmarks:
        landmarks = []
        if results.left_hand_landmarks:
            landmarks.append(results.left_hand_landmarks)
        if results.right_hand_landmarks:
            landmarks.append(results.right_hand_landmarks)
        return keypoints, landmarks
    else:
        return keypoints