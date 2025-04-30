from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.keypoint_extractor import extract_keypoints

app = Flask(__name__)
model = load_model('model/model_rnn.h5')
label_list = ['Bapak', 'Hai', 'Ibu', 'Kamu', 'Maaf',
              'Makan', 'Nama', 'Samasama', 'Terimakasih', 'Tidur']
max_seq_len = 50
sequence = []

cap = cv2.VideoCapture(1)
prev_time = time.time()

# Variabel untuk menyimpan prediksi dan akurasi terakhir
latest_prediction = ""
latest_confidence = 0.0

def gen_frames():
    global sequence
    global prev_time
    global latest_prediction
    global latest_confidence
    while True:
        success, frame = cap.read()
        if not success:
            break

        keypoints, landmarks = extract_keypoints(frame, return_landmarks=True)
        image_height, image_width, _ = frame.shape

        if len(landmarks) > 0:
            # Gambar bounding box untuk tiap tangan
            margin = 20  # Margin untuk memperbesar bounding box
            for hand_landmarks in landmarks:
                coords = [(int(lm.x * image_width), int(lm.y * image_height)) for lm in hand_landmarks.landmark]
                x_coords, y_coords = zip(*coords)
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)

                # Menambahkan margin pada bounding box
                x_min = max(x_min - margin, 0)
                y_min = max(y_min - margin, 0)
                x_max = min(x_max + margin, image_width)
                y_max = min(y_max + margin, image_height)

                # Gambar bounding box yang sudah disesuaikan
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Tambahkan keypoints ke sequence
            sequence.append(keypoints)

            if len(sequence) > max_seq_len:
                sequence = sequence[-max_seq_len:]

            if len(sequence) == max_seq_len:
                input_seq = pad_sequences([sequence], maxlen=max_seq_len, padding='post', dtype='float32')
                prediction = model.predict(input_seq, verbose=0)
                label = label_list[np.argmax(prediction)]
                confidence = np.max(prediction)

                # Simpan prediksi dan akurasi terbaru
                latest_prediction = label
                latest_confidence = confidence

        else:
            # Reset sequence dan tampilkan pesan No Gesture
            sequence = []
            latest_prediction = 'No gesture detected'
            latest_confidence = 0.0

        # Tampilkan FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Encode frame untuk stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/latest_prediction')
def latest_prediction_data():
    return jsonify(prediction=latest_prediction, confidence=latest_confidence * 100)


if __name__ == '__main__':
    app.run(debug=True)
