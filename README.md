# Sign Language Recognition using RNN

Proyek ini merupakan bagian dari Tugas Akhir dengan fokus pada **pengenalan Bahasa Isyarat Indonesia** menggunakan pendekatan **Recurrent Neural Network (RNN)**. Sistem ini dirancang untuk mengenali gerakan isyarat dalam bentuk **kata**. 

Dataset digunakan dalam bentuk **video gerakan isyarat**, yang kemudian diekstrak menjadi rangkaian titik kunci (keypoints) menggunakan **MediaPipe**. Model RNN kemudian dilatih untuk mengenali pola urutan dari keypoints tersebut.



## Keterbatasan

Proyek ini masih mengalami **keterbatasan dalam jumlah dan variasi dataset** video untuk tiap kelas kata. Jika Anda tertarik untuk **melanjutkan atau mengembangkan proyek ini**, Anda dapat:
- Menambahkan dataset video baru.
- Memasukkan video tersebut ke folder masing-masing kelas menggunakan `atur_folder.py`.
- Melatih ulang model dengan `train.py`.



## Cara Menjalankan Proyek

Berikut adalah urutan langkah untuk menjalankan proyek ini:

1. ### Ekstrak Keypoints
   Jalankan script berikut untuk mengekstrak keypoints dari video isyarat:
   ```bash
   python extract_keypoints.py
   ```
   Ini akan menyimpan data hasil ekstraksi ke dalam folder `MP_Data` secara terstruktur.

2. ### Atur dan Rapikan Folder
   Setelah keypoints diekstrak, gunakan script berikut untuk **memasukkan data secara otomatis ke masing-masing kelas**:
   ```bash
   python atur_folder.py
   ```

3. ### Latih Model
   Untuk melatih model RNN berdasarkan dataset yang tersedia:
   ```bash
   python train.py
   ```

4. ### Pengujian Realtime
   Untuk melakukan **pengujian model secara realtime** menggunakan webcam:
   ```bash
   python realtime_inference.py
   ```

5. ### Dashboard Aplikasi
   Untuk menjalankan **dashboard visualisasi dan kontrol** dari aplikasi:
   ```bash
   python app.py
   ```



## Kontribusi

Jika Anda ingin berkontribusi, Anda bisa:
- Menambahkan lebih banyak video isyarat ke dalam folder `Video_Data`.
- Menjalankan `extract_keypoints.py` dan `atur_folder.py` untuk memperbarui dataset.
- Melatih ulang model agar lebih akurat dan generalisasi lebih baik.
