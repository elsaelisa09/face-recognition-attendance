# Menjalankan aplikasi pengenalan wajah

Paket ini menambahkan pipeline CPU dan aplikasi Streamlit ke proyek yang sudah
memiliki galeri 70 identitas dan model InsightFace lokal. Gunakan Python 3.10+
dan environment `tugasakhir` yang sudah berhasil menjalankan eksperimen.

## 1. Salin berkas

Ekstrak ZIP ke folder sementara. Salin **isi** folder hasil ekstraksi ke root
proyek `C:\Users\elsae\Downloads\face-recognition`, sejajar dengan folder
`artifacts`, `dataset`, `notebooks`, dan `weights`. Jika `src` sudah ada,
gabungkan foldernya. Cadangkan berkas dengan nama sama sebelum menggantinya.

Berkas utama tambahan:

- `streamlit_app.py`: halaman unggah foto/kamera dan pilihan wajah.
- `src/face_pipeline.py`: pembacaan gambar, deteksi, alignment, embedding,
  normalisasi L2, pencocokan centroid, dan keputusan threshold.
- `check_face_app.py`: pemeriksaan pemuatan model dan inferensi CLI.
- `requirements-streamlit.txt`: rentang dependensi untuk reproduksi.
- `tests/test_face_pipeline.py`: pengujian logika menggunakan backend sintetis.

Paket tidak menyertakan foto pribadi, galeri biometrik, konfigurasi hasil
eksperimen, atau bobot model. Aplikasi membacanya dari berkas yang sudah kamu buat.

## 2. Pastikan berkas proyek tersedia

Lokasi yang dibutuhkan:

```
artifacts/inference_config.json
artifacts/gallery_70.npz
weights/insightface/models/buffalo_l/det_10g.onnx
weights/insightface/models/buffalo_l/w600k_r50.onnx
```

Konfigurasi dibaca memakai skema notebook `04_app_preparation.ipynb` pada
percakapan. Galeri harus cocok dengan `gallery_sha256`. Semua label NPZ harus
bertipe string Unicode agar dapat dibaca dengan `allow_pickle=False`.

## 3. Instal Streamlit

Buka Anaconda Prompt:

```bat
conda activate tugasakhir
cd /d C:\Users\elsae\Downloads\face-recognition
python -m pip install "streamlit>=1.40,<2"
```

Environment ini sudah memiliki InsightFace, Pillow, NumPy, dan ONNX Runtime yang
dipakai pada eksperimen. Gunakan `requirements-streamlit.txt` untuk membuat
environment lain, bukan untuk mengganti semua versi di environment yang sudah
berjalan. InsightFace 0.7.3 dapat memerlukan compiler saat dipasang dari source
di Windows; instalasi lokal yang sudah berhasil tidak perlu diulang.

## 4. Periksa pipeline

```bat
python check_face_app.py
```

Target: `Pipeline siap`, 70 identitas, dan input `(112, 112)`. Model dimuat langsung
dari dua ONNX lokal. Aplikasi tidak menjalankan pengunduhan otomatis.

Opsional, periksa satu foto penuh:

```bat
python check_face_app.py --image "C:\lokasi\foto.jpg"
```

Untuk beberapa wajah, keluaran berstatus `select_face` dengan nomor kotak dan
landmark. Pilih menggunakan `--face-index`, berdasarkan pratinjau atau keluaran
aplikasi. Untuk crop yang sudah disejajarkan 112×112, gunakan `--aligned`.

## 5. Buka Streamlit

```bat
python -m streamlit run streamlit_app.py
```

Buka alamat lokal yang tercetak pada terminal, biasanya `http://localhost:8501`.
Unggah foto, lihat kotak wajah, pilih target jika ada beberapa orang, lalu tekan
**Kenali wajah**. Kamera menghasilkan foto sekali ambil, bukan video real-time.
Nomor wajah berlaku untuk foto yang sedang dibuka, bukan nomor identitas orang.

Foto yang diunggah dan hasilnya disimpan sementara dalam memori sesi browser.
Kode ini tidak menulis unggahan atau hasil inferensi ke disk. Ketika memilih foto
lain, hasil sebelumnya pada sesi dihapus. Model dibagi lewat cache resource dan
pemanggilan backend dijaga lock untuk penggunaan lintas sesi.

## Batas evaluasi yang harus ikut ditulis pada portfolio

- Evaluasi closed-set: 279 foto dari 70 identitas; semua metode baseline yang
  dilaporkan menggunakan query yang dikeluarkan dari galeri.
- Uji ketahanan menggunakan perubahan pada crop: itu belum mengukur robustness
  detector, liveness, atau foto baru dari sesi pengambilan berbeda.
- Threshold berasal dari simulasi galeri 60 identitas dengan dua referensi per
  orang. Validasi: 59 foto known + 20 unknown dari lima identitas. Test: 60 foto
  known + 20 unknown dari lima identitas berbeda; hasil yang dilaporkan seluruhnya
  benar pada split tersebut.
- Galeri aplikasi berisi 70 identitas. Threshold transfer berstatus
  `provisional_for_70_identities`; penerimaan/penolakan pada galeri ini belum
  divalidasi menggunakan orang di luar 70 identitas. Status tersebut terlihat di
  aplikasi. Kalibrasikan dengan data pengembangan terpisah lalu uji dengan data
  baru sebelum mengganti status validasinya.
- Skor cosine bukan probabilitas identitas. Pemilihan wajah terbesar juga tidak
  membuktikan identitas; karena itu aplikasi meminta pilihan eksplisit jika ada
  beberapa wajah.
- Memakai foto referensi sendiri sebagai unggahan hanya memeriksa alur aplikasi;
  hasil itu tidak menjadi evaluasi generalisasi baru.

## Pemeriksaan sebelum hosting

1. Jalankan aplikasi lokal dengan foto satu wajah, beberapa wajah, tanpa wajah,
   file tidak valid, dan foto dari kamera. Periksa gambar alignment serta hasilnya.
2. Ukur waktu total yang mencakup pembacaan, deteksi, alignment, dan pengenalan.
   Waktu pemuatan model dilaporkan terpisah oleh CLI; `recognition_ms` tidak
   mencakup deteksi dan pembacaan file.
3. Sediakan dua ONNX, galeri, serta konfigurasi pada host Python tujuan. Kode ini
   belum mengunggah atau memublikasikan aplikasi. CPU/memori host perlu diuji.
4. Simpan foto, nama, galeri embedding, dan konfigurasi terkait secara privat.
   Paket ini tidak menambahkan data tersebut ke repository publik.
5. Tinjau lisensi model sebelum distribusi/hosting. Kode InsightFace berlisensi
   MIT, sedangkan pretrained model publiknya memiliki ketentuan riset nonkomersial.

## Validasi paket ini

Pengujian otomatis memakai centroid dan backend sintetis untuk memeriksa label
Unicode, normalisasi, keputusan threshold, tidak ada wajah, dan pemilihan wajah
ganda. Sintaks berkas Python diperiksa. Model asli, galeri pribadi, browser,
webcam, dan deployment belum dapat diuji di lingkungan pembuat paket. Pemeriksaan
CLI dan aplikasi lokal di atas menggunakan model dan data proyekmu yang asli.

Sumber teknis:

- https://github.com/deepinsight/insightface
- https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
- https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource
- https://docs.streamlit.io/develop/api-reference/widgets/st.camera_input

