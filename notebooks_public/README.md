# Notebook Publik: Reproduksi Workflow

Folder ini berisi **salinan kode** yang telah dirapikan dari notebook eksperimen asli. Notebook asli dan hasil eksekusinya tetap disimpan secara privat di `notebooks/`. Salinan publik tidak memuat foto, nama peserta, embedding, output cell, atau metadata sesi penulis.

## Yang dapat dan tidak dapat direproduksi

- **Dapat diulang:** audit, standardisasi, alignment, ekstraksi embedding, perbandingan metode, robustness, split open-set, kalibrasi, dan ekspor galeri menggunakan data sendiri.
- **Tidak dijanjikan sama:** angka akurasi, threshold, manifest, hash, dan galeri eksperimen awal. Dataset mahasiswa tidak dibagikan.
- **Belum tersedia sebagai alur umum:** inisialisasi schema Supabase dan bootstrap galeri baru dari jumlah identitas arbitrer. Importer aplikasi awal masih mengasumsikan 70 identitas. Notebook tidak menulis ke database.

## Menjalankan

1. Gunakan checkout proyek lengkap, bukan folder notebook saja. Jalankan dari root proyek atau `notebooks_public/`.
2. Pasang dependensi melalui `python -m pip install -r requirements-research.txt`, lalu buka `python -m jupyter lab` pada environment yang sama.
3. Siapkan data sendiri di `dataset/raw/<label_orang>/<foto>`. Gunakan 3–4 foto berbeda per identitas untuk protokol ini; empat foto menyediakan dua referensi, satu validasi, dan satu test pada identitas known. Label pseudonim disarankan.
4. Jalankan notebook sesuai urutan tabel di bawah. Unduhan model pertama memerlukan internet dan mengikuti ketentuan model penelitian nonkomersial.
5. Di notebook 01, review foto multiwajah dilakukan **secara manual**. Aktifkan `SHOW_PRIVATE_PREVIEWS`, panggil `review_next()`, lihat foto, lalu pilih nomor melalui `pilih_wajah(nomor)`. Jangan menyalin pilihan wajah contoh tanpa melihat foto.
6. Notebook 03 secara default menyisihkan **5 identitas unknown validasi + 5 unknown test**, sehingga membutuhkan minimal 12 identitas. Untuk data lebih kecil, ubah kedua parameter sambil menyisakan minimal dua identitas known dan foto validation known yang cukup.
7. Evaluasi threshold di test setelah dibekukan. Jangan memilih ulang threshold berdasarkan hasil test.

| Notebook | Hasil utama |
|---|---|
| [01_data_audit.ipynb](01_data_audit.ipynb) | Manifest, foto RGB terstandardisasi, pilihan wajah, alignment 112 × 112 |
| [02_model_comparison.ipynb](02_model_comparison.ipynb) | Embedding 512 dimensi, few-shot 30 pembagian, robustness |
| [03_open_set_evaluation.ipynb](03_open_set_evaluation.ipynb) | Split, galeri kalibrasi, threshold, prediksi dan metrik test |
| [04_app_preparation.ipynb](04_app_preparation.ipynb) | Galeri seluruh identitas dan konfigurasi inferensi lokal |

Sel percobaan lama, reset pilihan review, dan pemilihan wajah yang otomatis di-hardcode tidak dibawa ke salinan publik. Jumlah foto/identitas pada pemeriksaan dan nama artefak dibuat dinamis. Kebijakan kalibrasi, 30 seed few-shot, dan transformasi robustness tetap mengikuti eksperimen asli. Simpan dataset dan artefak setiap eksperimen terpisah jika mengubah sumber data atau protokol.

## Status pemeriksaan

Source diperiksa sintaksnya. Split, kalibrasi, verifikasi galeri, test, dan ekspor artefak diperiksa menggunakan embedding sintetis pada konfigurasi 70 dan 12 identitas. Pemeriksaan sintetis memastikan konsistensi kode, bukan mengukur akurasi wajah. **Notebook publik belum dieksekusi ulang end-to-end dengan foto asli, model ONNX, atau database Supabase dalam revisi ini.**

## Menjaga output tetap privat

Menjalankan notebook dapat membuat output wajah atau nama muncul lagi. Sebelum commit:

```bash
python scripts/clear_notebook_outputs.py
git diff -- notebooks_public
git status
```

Script hanya menyentuh salinan di `notebooks_public/`. Periksa juga data yang mungkin ditulis langsung di source, Markdown, screenshot, checkpoint, serta file tambahan. Jangan commit `dataset/`, `reports/`, embedding NPZ, bobot model, Secrets, atau backup database.

Pertahankan aturan `.gitignore` untuk `notebooks/` asli. Folder `notebooks_public/` menggunakan nama berbeda sehingga tidak perlu membuka seluruh folder notebook privat. Pastikan aturan `.gitignore` tidak mengabaikan **semua** `*.ipynb` atau folder `notebooks_public/`.

Petunjuk aplikasi, Supabase, dan deployment tersedia pada [SETUP.md](../SETUP.md).
