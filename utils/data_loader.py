import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(data_dir, max_seq_len=None):
    X, y, label_map = [], [], {}
    label_list = sorted(os.listdir(data_dir))

    for label_idx, label in enumerate(label_list):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            files = [f for f in os.listdir(label_path) if f.endswith('.npy')]
            if not files:
                continue
            for file in files:
                sample = np.load(os.path.join(label_path, file))
                X.append(sample)
                y.append(label_idx)
            label_map[label_idx] = label

    if not X or not y:
        return np.array([]), np.array([]), {}, []

    # Menentukan max_seq_len jika tidak diberikan
    if max_seq_len is None:
        max_seq_len = max([x.shape[0] for x in X])

    # Padding sequences
    X = pad_sequences(X, maxlen=max_seq_len, dtype='float32', padding='post')

    return np.array(X), np.array(y), label_map, label_list

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
