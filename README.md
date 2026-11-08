# Inspeksi Visual Komponen — FGD Kelompok 4

Aplikasi Streamlit ini menggunakan artefak eksperimen terbaru dalam folder ini untuk memeriksa `screw` dan `metal_nut`. Tiga halaman tersedia: AnomalyDINO (1-shot, 16-shot, full-shot), EfficientAD, dan perbandingan prediksi pada gambar yang sama.

## Menjalankan secara lokal

Gunakan Python 3.12 dari direktori ini:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Setiap kategori dan kandidat mempunyai batas gambar serta piksel sendiri. `deployment_config.json` memakai nilai dari `ringkasan_threshold.csv` di folder ini. Model tidak dilatih ulang saat aplikasi dijalankan.

## Artefak dan kode model

- Bobot backbone: `dinov2_vits14_state_dict.pt` (DINOv2 ViT-S/14).
- Memory bank AnomalyDINO: `screw/anomalydino/` dan `metal_nut/anomalydino/`.
- Checkpoint EfficientAD 60 epoch: `screw/efficientad/` dan `metal_nut/efficientad/`.
- Komponen kode AnomalyDINO yang dipakai notebook: `vendor/AnomalyDINO/`, commit `b9d1c2648e3a5247437d4d953d907a8f3d994457`.
- Arsitektur DINOv2 lokal: `vendor/dinov2/`, commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.

Repo model disertakan agar aplikasi tidak perlu mengunduhnya saat mulai. Aplikasi mengikuti alur inferensi adaptasi notebook, termasuk RGB, resize DINOv2 448, masking query sesuai kategori, pencarian tetangga terdekat pada bank, serta preprocessing EfficientAD 256. Keputusan `CACAT` dibuat saat skor gambar lebih besar dari batas gambar dalam konfigurasi. Mask prediksi memakai batas piksel kandidat tersebut.

Folder ini belum berisi tabel metrik test yang dapat dipastikan berasal dari checkpoint dan threshold terbaru. Karena itu halaman perbandingan menampilkan prediksi keempat kandidat tanpa mengklaim model terbaik menurut metrik run lain.

## Streamlit Community Cloud

Gunakan `app.py` sebagai entrypoint. Berkas `*.faiss`, `*.pt`, dan `*.ckpt` memakai Git LFS melalui `.gitattributes`. Pastikan Git LFS aktif dan semua berkas besar benar-benar terunggah sebelum deploy. Aplikasi memuat bank yang dipilih saja; full-shot tetap dapat memerlukan RAM besar dan berjalan lambat pada CPU.
