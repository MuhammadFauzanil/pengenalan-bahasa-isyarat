import os
import shutil

# Path ke folder preprocessed kamu
data_dir = "data/preprocessed"

# Loop semua file dalam folder
for file in os.listdir(data_dir):
    if file.endswith(".npy"):
        # Ambil label dari nama file, misal 'Aku_1.npy' → 'Aku'
        label = file.split("_")[0]
        label_dir = os.path.join(data_dir, label)

        # Buat folder label jika belum ada
        os.makedirs(label_dir, exist_ok=True)

        # Pindahkan file ke folder label
        src = os.path.join(data_dir, file)
        dst = os.path.join(label_dir, file)
        shutil.move(src, dst)
        print(f"✅ {file} dipindahkan ke {label}/")
