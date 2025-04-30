from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.data_loader import load_data, split_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.layers import Input


print('🔍 Loading data...')
X, y, label_map, label_list = load_data('data/preprocessed', max_seq_len=50)
X_train, X_test, y_train, y_test = split_data(X, y)

print('🚀 Training...')
model = Sequential()
model.add(Input(shape=(50, 126)))
model.add(Masking(mask_value=0., input_shape=(50, 126)))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_map), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

print(f"Shape X: {X.shape}")  # Seharusnya (jumlah_data, 50, 225)
print(f"Shape y: {y.shape}")


print('💾 Saving model...')
model.save('model/model_rnn.h5')
print('✅ Training selesai. Model disimpan di model/model_rnn.h5')

# 🔍 Evaluasi model
print('📊 Evaluasi model pada data uji...')
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Akurasi
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Akurasi: {acc:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("hasil_confusion_matrix.png")  # ✅ Simpan sebagai PNG
print("🖼 Confusion matrix disimpan sebagai 'hasil_confusion_matrix.png'")
plt.close()


# Classification Report
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_list))