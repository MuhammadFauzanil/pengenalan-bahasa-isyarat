import cv2
import mediapipe as mp
import numpy as np
import os

from utils.keypoint_extractor import extract_keypoints

input_dir = 'data/videos'
output_dir = 'data/preprocessed'

os.makedirs(output_dir, exist_ok=True)

for label in os.listdir(input_dir):
    label_path = os.path.join(input_dir, label)
    for file in os.listdir(label_path):
        cap = cv2.VideoCapture(os.path.join(label_path, file))
        sequence = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            keypoints = extract_keypoints(frame)
            sequence.append(keypoints)
        cap.release()
        np.save(f"{output_dir}/{label}_{file.split('.')[0]}.npy", sequence)

print('✅ Ekstraksi selesai, cek folder data/preprocessed')