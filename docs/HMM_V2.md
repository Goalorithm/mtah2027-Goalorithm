# Kandidat HMM v2: batas usia bukti posisi bola

Peningkatan 11 November 2026. Versi v1 dan config default tetap tersedia. V2 adalah kandidat
perbaikan perilaku model; belum dipilih sebagai pemenang berdasarkan akurasi.

## Masalah dan perubahan

V1 menggunakan posisi bola terakhir di luar lapangan sebagai bukti untuk label D sampai ada
observasi baru. Ketika tracking hilang lama, bukti geometris itu dapat bertahan tanpa batas usia.
V2 memakai `hmm.outside_memory_frames = 50` di `config/exp_hmm_v2.toml`.

Pada observasi baru, bobot bukti luar lapangan tetap penuh. Untuk observasi berumur `age` frame,
pengalinya menjadi `max(0, 1 - age / outside_memory_frames)`. Saat usia mencapai 50 frame
(2 detik pada 25 fps), bukti geometris itu menjadi nol. Nol frame berarti hanya mempercayai
observasi saat ini. Jika key tersebut tidak ada, perilaku lama v1 dipertahankan persis.

Menghapus bukti lama tidak memaksa label H atau A. Model masih dapat memilih D berdasarkan
kehilangan tracking, gerak pemain, bukti lain, dan konteks urutan. Riwayat/interpolasi mentah
tidak diubah dan tetap terpisah antar babak. Tidak ada koreksi label berdasarkan tinjauan video.

50 frame adalah asumsi awal yang eksplisit dan dapat diuji; bukan angka hasil tuning pada
ground truth publik, pseudo-label sendiri, maupun label manusia pertandingan ini.

## Pemeriksaan data asli tanpa ground truth

| Pemeriksaan | V1 | V2 |
| --- | ---: | ---: |
| Possession yang diprediksi | 237 | 238 |
| Possession di bawah 2 detik | 6 | 6 |
| Proporsi D pada bagian dinilai | 36,463% | 36,292% |
| Rata-rata durasi possession | 17,035 detik | 17,009 detik |
| Observasi luar lapangan yang sudah kedaluwarsa | 8.891 frame | 8.891 frame |
| Bukti geometris tersisa pada frame tersebut | 8.891 frame | 0 frame |
| Label D pada frame tersebut | 8.891 frame | 8.620 frame |

334 dari 158.857 frame yang dinilai berubah: 131 D ke H, 140 D ke A, dan 63 A ke H.
Bagian di luar penilaian tidak berubah. Pergeseran juga dapat terjadi sebelum/sesudah frame
bukti berubah karena Viterbi memilih jalur urutan secara bersama.

Laporan awal memakai umur lebih dari 50 frame dan menghitung 8.875 frame. Laporan v2 memakai
umur minimal 50 frame, sesuai waktu bukti habis, sehingga ada 8.891 frame. Ini perbedaan
definisi batas, bukan pertambahan kasus atau perubahan data.

Angka ini bukan pengukuran akurasi dan tidak membuktikan 334 perubahan label adalah benar.
Tidak ada skor publik v2 yang dipakai untuk memilih parameter. Evaluasi independen yang
diizinkan tetap diperlukan; lihat [RULE_UPDATE.md](RULE_UPDATE.md).

## Reproduksi

```powershell
python -m mtah.predict --config config/exp_hmm_v1.toml --name exp_hmm_v1_compat --no-score
python -m mtah.predict --config config/exp_hmm_v2.toml --name exp_hmm_v2 --no-score
python tools/hmm_reliability_report.py
python tools/ai_labels.py --config config/exp_hmm_v2.toml --out-dir outputs/ai_labels_v2
```

Di Windows, `label_ai_v2.cmd` menjalankan ekspor AI v2. Berkas v1 di `outputs/ai_labels/`
dipertahankan; hasil baru berada di `outputs/ai_labels_v2/`. Keduanya merupakan pseudo-label,
bukan ground truth independen. V2 mencakup keenam jendela, 45.000 frame, dan 108 titik label.

SHA-256 prediksi:

- V1, tetap identik: `48f403095696ac83a60a06787c1320eff2db5c8ca95a7f81ffc8c44752773e86`.
- V2: `3d8bba74bc0ac276f5d370866abd58dfe974b7efa5aa17757ee212b6b2733078`.

Laporan diagnostik: `outputs/tech1_review/hmm_v2_reliability.json`. Manifest sumber, config,
dan output: `outputs/ai_labels_v2/manifest.json`. Semua hasil tetap di-ignore Git.

## Pengujian dan batas

Tes mencakup kedaluwarsa bukti tanpa mengubah riwayat mentah, observasi baru yang tetap dipercaya,
mode observasi saat ini saja, reset antar babak, hilangnya bola selama ribuan frame, stoppage
panjang yang tetap D setelah bukti geometri habis, dan rangkaian sentuhan singkat, pergantian
kontrol, stoppage, serta restart. Tes sintetis memeriksa perilaku yang dituju, bukan akurasi laga.

Model belum menyaring seluruh outlier tracking mentah; observasi salah yang baru masuk masih
dapat memengaruhi prediksi. Batas umur ini hanya mengatasi pengaruh observasi lama. Parameter
lain dan kelemahan nearest-player geometry tetap sama dengan v1.
