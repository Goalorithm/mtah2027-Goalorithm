# Panduan membuat label validasi (anggota bisnis)

> **DIARSIPKAN, 11 November 2026.** Aturan resmi terbaru melarang label manusia pada pertandingan
> yang disediakan, termasuk untuk validasi dan kalibrasi berlabel. Jangan menjalankan langkah
> anotasi di bawah pada BAR-ATH. Gunakan [RULE_UPDATE.md](RULE_UPDATE.md) dan `tools/ai_labels.py`.
> Isi lama dipertahankan hanya sebagai riwayat panduan tim sebelum perubahan aturan.

> Anggota bisnis tidak memakai GitHub. Versi yang mereka baca adalah Google Doc
> **"Panduan Label Validasi – Goalorithm"** di folder Drive tim, dengan isi yang sama ditambah
> cara memasang mpv tanpa command line. Label diisi di spreadsheet **"Goalorithm – Sheet Tim"**
> (tab `windows`, `perubahan`, `Tugas tim bisnis`). Kalau salah satu diubah, samakan yang lain.

Label ini adalah **alat ukur tim**. Dengan label ini tim tahu apakah sebuah ide benar-benar
menaikkan skor, tanpa bergantung pada 5 menit ground truth panitia yang terlalu kecil untuk
memilih. Label manual **tidak pernah** masuk ke file submission, karena aturan panitia
mewajibkan label submission keluar dari algoritma. Memakai label manual untuk validasi
diperbolehkan.

## 1. Persiapan (sekali)

1. Unduh `BAR-ATH_tactical.mp4` (4,86 GB) dari folder `tactical` di Google Drive panitia ke
   laptopmu. Jangan dibagikan ulang ke tempat lain.
2. Pasang **mpv** dari https://mpv.io/installation/ (bagian Windows). mpv dipakai karena bisa
   menampilkan nomor frame dan maju-mundur satu frame.
3. Buka video dengan nomor frame di pojok kiri atas. Di PowerShell:

```powershell
mpv --osd-level=3 --osd-status-msg='frame ${estimated-frame-number}' "C:\path\ke\BAR-ATH_tactical.mp4"
```

Untuk langsung lompat ke frame tertentu, tambahkan `--start=<detik>` dengan detik = frame ÷ 25.
Contoh untuk frame 30000: `--start=1200`.

| Tombol | Fungsi |
| --- | --- |
| Spasi | Play / pause |
| `.` | Maju 1 frame (otomatis pause) |
| `,` | Mundur 1 frame |
| ← / → | Mundur / maju 5 detik |
| `[` / `]` | Pelankan / cepatkan pemutaran |

## 2. Arti label

| Label | Kapan |
| --- | --- |
| `H` | Barcelona menguasai bola, bola hidup |
| `A` | Athletic Club menguasai bola, bola hidup |
| `D` | Bola mati: keluar lapangan, peluit pelanggaran, gol, sampai bola dimainkan lagi |

Aturan dasar dari panitia (CHALLENGE.md):

- Possession baru berpindah kalau lawan **mengontrol** bola. Sentuhan tanpa kontrol, defleksi,
  atau menang duel udara tanpa menguasai bola sesudahnya **tidak** memindahkan possession.
- Saat bola liar atau direbutkan, possession tetap milik tim yang terakhir menguasai.
- Ground truth panitia dibuat dari **data tracking yang sudah dikoreksi**, bukan dari orang yang
  menonton. Detail seperti "kapan tepatnya bola dianggap mati" dipelajari dari kalibrasi di
  bawah dan dicatat di [ATURAN_POSSESSION.md](ATURAN_POSSESSION.md).

## 3. Kalibrasi dulu (wajib, sebelum jendela lain)

Kalibrasi memakai jendela publik, **frame 11185 sampai 18684** (video 7:27 sampai 12:27).

1. **Jangan** melihat ground truth publik atau gambar timeline-nya dulu.
2. Label jendela itu seperti di langkah 4, dengan nama jendela `PUBLIK`.
3. Minta Tech 1 menjalankan:

```powershell
python tools/labels_to_csv.py --labeler NamaKamu --out labels/pelabel_namakamu.csv
python tools/timeline.py --window publik --pred labels/pelabel_namakamu.csv baseline_smooth50
```

4. Bandingkan gambar timeline-nya bersama tim. Setiap perbedaan dengan ground truth adalah
   pelajaran tentang cara panitia mendefinisikan possession. Catat di ATURAN_POSSESSION.md.

## 4. Cara melabeli satu jendela

Satu jendela = 7500 frame (5 menit). Perkiraan waktunya 1–2 jam.

Yang dicatat **hanya titik perubahan**, di spreadsheet "Goalorithm – Sheet Tim", tab `perubahan`:

| jendela | frame_mulai | label | pelabel | catatan |
| --- | --- | --- | --- | --- |
| V1 | 30000 | H | Sari | awal jendela |
| V1 | 30212 | A | Sari | intersep di lini tengah |
| V1 | 30540 | D | Sari | bola keluar di sisi kanan |
| V1 | 30690 | A | Sari | lemparan ke dalam Athletic |

(Contoh di atas karangan, bukan data asli.)

Langkahnya:

1. Lompat ke `frame_awal` jendela (dari tab `windows`), lalu tulis baris pertama **tepat** di
   frame itu dengan keadaan saat itu.
2. Putar video. Setiap kali keadaan berubah, mundurkan per frame (`,`) sampai frame pertama
   keadaan baru, lalu catat frame itu.
3. Berhenti di `frame_akhir`. Tidak perlu baris penutup.
4. Kolom `catatan` opsional tapi sangat membantu tim tech untuk memahami kasus sulit.

Ketelitian yang dibutuhkan: di sekitar titik perubahan usahakan selisih 2–3 frame. Selisih 1 detik
sudah menurunkan skor cukup banyak.

## 5. Pembagian dan pengumpulan

- Tech 1 mengisi tab `windows` (6 jendela, 4 `dev` + 2 `holdout`) dan siapa pelabelnya.
- Satu jendela dilabeli **kedua** anggota bisnis secara terpisah, untuk mengukur seberapa sama
  hasil dua orang. Kalau sangat berbeda, aturannya perlu dibahas sebelum lanjut.
- Selesai satu jendela: ubah status di tab `Tugas tim bisnis` dan kabari di grup WhatsApp.
  Tech 1 mengunduh tab `windows` dan `perubahan` sebagai CSV ke `labels/windows.csv` dan
  `labels/perubahan.csv`, lalu menjalankan `python tools/labels_to_csv.py`. Kalau ada kesalahan
  format, skrip menunjuk nomor baris di spreadsheet yang harus diperbaiki.
