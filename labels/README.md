# Area diagnostik dan label AI

Aturan resmi terbaru melarang label manusia pada pertandingan BAR-ATH, termasuk untuk training
dan validasi. Alur anotasi manusia sebelumnya sudah diarsipkan. Lihat
[pembaruan aturan](../docs/RULE_UPDATE.md).

- `windows.csv` memuat jendela publik dan enam area diagnostik untuk pseudo-label algoritmik
  (`tools/ai_labels.py`). Tidak ada orang yang melabeli area itu.
- `perubahan.csv` tetap kosong; jangan isi dengan anotasi manusia atau salinan pseudo-label.
- `validasi_dev.csv` dan `validasi_holdout.csv` tidak dibuat dari pseudo-label AI.
- `python tools/ai_labels.py` menghasilkan label AI di folder ignored `outputs/ai_labels/`.

Hasil AI mencakup 45.000 frame dan 108 titik awal/perubahan. Kolom `label_ai` dan `anotator_ai`
membedakannya dari format anotasi manusia. Manifest memuat hash data, config, kode, dan hasil.
Nama kelompok dev/holdout hanya identitas jendela lama, bukan split validasi independen.

Label berasal dari HMM yang sedang dikembangkan. Jangan memakai label tersebut untuk mengukur
akurasi HMM terhadap dirinya sendiri. Ketidaksetujuan dengan baseline adalah diagnostik,
bukan ground truth atau estimasi probabilitas salah.
