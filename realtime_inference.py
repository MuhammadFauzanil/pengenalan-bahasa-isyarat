import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.keypoint_extractor import extract_keypoints

model = load_model('model/model_rnn.h5')
label_list = ['Bapak', 'Hai', 'Ibu', 'Kamu', 'Maaf',
              'Makan', 'Nama', 'Samasama', 'Terimakasih', 'Tidur']

max_seq_len = 50
sequence = []

cap = cv2.VideoCapture(1)
print('🎥 Real-time inference berjalan... (Tekan Q untuk keluar)')

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    start_time = time.time()

    keypoints, hand_landmarks = extract_keypoints(frame, return_landmarks=True)
    sequence.append(keypoints)

    # Kotak tangan berdasarkan landmark
    if hand_landmarks:
        for hand in hand_landmarks:
            x_list = [lm.x for lm in hand.landmark]
            y_list = [lm.y for lm in hand.landmark]
            h, w, _ = frame.shape
            x_min = int(min(x_list) * w) - 10
            y_min = int(min(y_list) * h) - 10
            x_max = int(max(x_list) * w) + 10
            y_max = int(max(y_list) * h) + 10
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Prediksi jika sequence cukup panjang
    if len(sequence) > max_seq_len:
        sequence = sequence[-max_seq_len:]

    if len(sequence) == max_seq_len:
        input_seq = pad_sequences([sequence], maxlen=max_seq_len, padding='post', dtype='float32')
        prediction = model.predict(input_seq, verbose=0)
        label = label_list[np.argmax(prediction)]
        confidence = np.max(prediction)
        cv2.putText(frame, f'{label} ({confidence:.2f})', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Sign Language Recognition', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()