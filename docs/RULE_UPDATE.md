# Pembaruan aturan dan label AI

Diperiksa 11 November 2026. Repo tim masih memakai dokumen lama; pembaruan sumber resmi
berada pada commit upstream `f8f2629ff1e0985f81d7e59d8413678514ee1530`.

Sumber resmi yang diperiksa:

- [RULES.md pada commit tersebut](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/RULES.md),
  bagian *Training and validation data* dan *What remains fine on the provided game*.
- [FAQ.md pada commit tersebut](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/FAQ.md),
  bagian penggunaan LLM dan pelabelan manual.

## Batas publikasi branch saat ini (17 November 2026)

Training dan kalibrasi model dibatasi pada pertandingan eksternal. Training HMM pada
tracking BAR-ATH, detektor bola dari video pertandingan tersebut, simulator turunannya,
dan bobot/CSV terkait telah dikeluarkan dari snapshot terbaru. Daftar pengecualian berada di
`.github/excluded-paths.txt`. Rincian izin pada catatan 11 November di bawah adalah catatan
historis, bukan dasar untuk mengaktifkan kembali jalur yang dikeluarkan.

## Perubahan yang berlaku

Label manusia pada pertandingan BAR-ATH tidak diperbolehkan untuk submission, training,
kalibrasi berlabel, maupun validasi. Label manusia dari pertandingan lain diperbolehkan dengan
asal data yang dapat diakses panitia. Pada branch ini, fitting supervised maupun unsupervised memakai data eksternal.
Pseudo-label inferensi lokal bukan acuan untuk fitting, kalibrasi, atau pemilihan model.

## Klarifikasi resmi panitia (email 3 November 2026)

Email panitia *"Clarification of the rules for the Initial Challenge"* menyatakan bahwa aturannya
mengesampingkan dokumentasi, email sebelumnya, dan jawaban individual.

Tidak diperbolehkan:

- memeriksa output terhadap video;
- label tangan, data buatan manusia, atau data lain yang dibuat manual tentang game ini;
- mengunggah data yang disediakan ke layanan AI;
- training, fitting, atau kalibrasi algoritma pada data ground-truth possession atau video
  yang disediakan.

Diperbolehkan: pemeriksaan otomatis yang hanya memakai data yang disediakan; menonton atau
memeriksa data dan video untuk memahami definisi possession; memakai video, tracking, Smart
Tagging, atau data hasil algoritma lain sebagai input selama algoritma tidak dilatih pada ground
truth atau video yang disediakan; menyertakan berkas tahap antara; serta memakai dataset yang
hak pakainya dimiliki. Training/retraining dan tuning parameter hanya memakai data eksternal.

Implementasi ini berjalan dengan komputasi lokal dan tidak mengunggah materi pertandingan ke
layanan AI. Alat peninjauan video dan saran label manual (`tools/suggest_labels.py`,
`src/mtah/video.py`, penyimpanan tinjauan di `src/mtah/review.py`, `docs/PANDUAN_LABEL.md`)
dikeluarkan dan dicatat di `.github/excluded-paths.txt`.

Arahan sebelumnya untuk menyelesaikan enam jendela dengan anotasi manusia sudah tidak berlaku.
Sumber panitia di working tree tidak ditimpa, dan perubahan pengguna pada `RULES.md` dipertahankan.
Sinkronisasi upstream penuh harus melalui branch/PR terpisah karena juga dapat mengubah kode panitia.

## Hasil label AI yang dibuat

Jalankan dari root repo:

```powershell
.venv\Scripts\python.exe -X utf8 tools/ai_labels.py
```

Di Windows bisa klik dua kali `label_ai.cmd`. Tidak perlu nama pelabel manusia atau tinjauan
manual. Proses menjalankan ulang model HMM + Viterbi dari tracking dan Smart Tagging asli,
kemudian mengekspor label algoritma pada enam jendela yang sebelumnya direncanakan tim.
Tidak ada label acuan, usulan lama, CSV prediksi lama, atau hasil penilaian yang menjadi input.

Hasil di `outputs/ai_labels/`, seluruhnya di-ignore Git:

| Berkas | Isi |
| --- | --- |
| `fullmatch_ai.csv` | Prediksi algoritma 199.319 frame; X di luar bagian yang dinilai |
| `perubahan_ai.csv` | 108 titik awal/perubahan label H/A/D di enam jendela |
| `intervals_ai.csv` | Interval dengan label HMM, label baseline, dan tanda perbedaan metode |
| `pseudolabel_dev.csv` | 30.000 frame pada empat jendela; X di tempat lain |
| `pseudolabel_holdout.csv` | 15.000 frame pada dua jendela; X di tempat lain |
| `manifest.json` | Asal algoritma, hash data/config/kode/output, statistik, dan batas penggunaan |

Nama `dev`/`holdout` dipertahankan sebagai identitas area lama. Berkas tersebut **bukan** split
validasi independen. Label `H` = Barcelona, `A` = Athletic, `D` = bola mati. Ketidaksetujuan
dengan baseline smooth50 menjadi penanda ketidakpastian, bukan probabilitas salah. Kedua
metode memakai tracking yang sama; kesepakatan juga tidak membuktikan kebenaran.

## Batas penilaian

Pseudo-label berasal dari HMM yang sedang diuji. Mengukur HMM terhadap pseudo-labelnya sendiri
akan menghasilkan pengukuran melingkar. Karena itu tidak dibuat `labels/validasi_dev.csv`,
tidak dihitung klaim akurasi dari pseudo-label, dan tidak dipilih pemenang model berdasarkan itu.
Kode scoring utama kini hanya menemukan contoh publik resmi untuk pemeriksaan kebenaran
pipeline. Contoh publik tidak menjadi dasar training, tuning, atau pemilihan versi.

Perbandingan akurasi yang sah memerlukan dataset berlabel dari pertandingan lain yang izinnya
sesuai atau evaluasi panitia. Training/evaluasi lengkap IDSSE sudah tersedia dengan provenance
dan batas transfer di [SUPERVISED_IMPROVEMENT.md](SUPERVISED_IMPROVEMENT.md).
Metode Tech 2/3 kini telah diintegrasikan lokal; detailnya di
[README tim](../README.md). Reproduksi lintas laptop, review PR, dan rilis tetap
dapat dikerjakan tanpa membuat label manusia pertandingan ini.

## Alur lama

Peluncur peninjauan manual dan pemilihan dev BAR-ATH sekarang memberi pesan bahwa alurnya
diarsipkan. Jalur default pembuatan/impor label manual dicegah, dan scoring tidak menemukan
berkas validasi lokal sebagai ground truth. Helper format dan peninjau sintetis dipertahankan
untuk pengujian; keberadaannya bukan izin membuat label manual BAR-ATH.
