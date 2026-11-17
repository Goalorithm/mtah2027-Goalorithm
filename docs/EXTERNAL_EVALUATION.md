# Evaluasi eksternal pada titik awal umpan

Catatan hasil 13 November 2026, disunting 17 November setelah pengecualian model.
Angka historis berikut bukan verifikasi checkpoint dengan kode terbaru.
Bangun ulang laporan setelah perubahan sumber.

 Evaluasi ini menambah acuan independen dari dua sample game
Metrica, tanpa membuat label manusia pada pertandingan BAR-ATH.

## Sumber dan batas acuan

Data berasal dari [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data),
commit `e706dd506b360d69d9d123d5b8026e7294b13996`. README sumber menyediakan tracking/event
pertandingan anonim yang tersinkron, koordinat ternormalisasi dan ukuran lapangan 105 x 68 m,
serta meminta penggunaan bertanggung jawab dan pengakuan sumber. Seluruh berkas dipin dengan
SHA-256 di `config/external_benchmark.toml` dan disimpan di ignored `data/external/metrica_sample/`.

Acuan dibuat secara mekanis dari tim pada event PASS di frame awal event. Ini asumsi benchmark
kontrol tim pada awal umpan, bukan anotasi H/A/D lengkap. Ketepatan semantik frame event dapat
berbeda dari batas kontrol pada ground truth possession. Timestamp, frame ID dan periodenya
harus cocok persis dengan tracking; offset tidak dicari agar hasil terlihat lebih baik.

- Sample Game 1: 145.006 frame; 799 event umpan menjadi 798 frame unik, tanpa konflik tim.
- Sample Game 2: 141.156 frame; 964 event umpan pada 964 frame unik, tanpa konflik tim.
- Cakupan acuan hanya 0,55% dan 0,68% frame. Tidak tersedia acuan bola mati atau seluruh
  perubahan possession, sehingga spellF1 pertandingan tidak dihitung.

H/A di benchmark berarti Home/Away sample game, bukan identitas Barcelona/Athletic. Tidak ada
event BAR-ATH dari pihak lain yang digunakan. Event eksternal hanya dibaca evaluator; model
menerima tracking dan daftar Smart Tagging kosong. Tidak ada fitting atau tuning kandidat di
benchmark ini; config dan checkpoint yang telah ada dipakai sama pada kedua game.

## Adapter tracking

CSV Home/Away harus memiliki frame yang berurutan dari 1, timeline identik, dan langkah waktu
25 fps. Frame ID 1 dipetakan ke indeks array 0. Kedua salinan koordinat bola diperiksa; jika
keduanya tersedia tetapi berbeda, proses berhenti. Jumlah slot pemain dapat berbeda antar tim.

Koordinat dinormalisasi sumber dipetakan secara affine ke konvensi lapangan model 105,3 x 68 m.
Kecepatan pemain diturunkan dari selisih koordinat per frame; perhitungan di-reset saat babak
berganti. Tiap babak di-decode terpisah sehingga voting, interpolasi dan state tidak menyeberang
batas babak. Perbedaan sumber kecepatan dan tidak adanya tagging merupakan pergeseran domain.

## Hasil konfigurasi yang dibekukan

Angka berikut adalah ketepatan tim **hanya pada frame awal umpan**:

| Model | Sample Game 1, 798 anchor | Sample Game 2, 964 anchor |
| --- | ---: | ---: |
| Baseline smooth50 | 749 benar / 93,86% | 928 benar / 96,27% |
| HMM v2, bobot rancangan | 758 benar / 94,99% | 929 benar / 96,37% |
| Tech 2: bola keluar | 735 benar / 92,11% | 916 benar / 95,02% |
| Tech 2: + restart | 735 benar / 92,11% | 916 benar / 95,02% |
| Tech 3: kontrol | 680 benar / 85,21% | 858 benar / 89,00% |
| Tech 3: + X | 680 benar / 85,21% | 858 benar / 89,00% |
| Tech 2+3: rules v1 | 681 benar / 85,34% | 856 benar / 88,80% |
| Supervised v1 | 771 benar / 96,62% | 929 benar / 96,37% |
| Supervised v2, tahan noise | 755 benar / 94,61% | 913 benar / 94,71% |
| Supervised v3, interaksi | 747 benar / 93,61% | 892 benar / 92,53% |
| Supervised v4, kapasitas lebih besar | 744 benar / 93,23% | 893 benar / 92,63% |
| Supervised boundary | 740 benar / 92,73% | 885 benar / 91,80% |
| Supervised possession-balanced | 744 benar / 93,23% | 890 benar / 92,32% |
| Joint terpilih pada validasi IDSSE | 740 benar / 92,73% | 885 benar / 91,80% |

Prediksi D/X pada anchor dihitung sebagai salah, bukan dibuang dari denominator. Event ganda
pada frame dan tim yang sama hanya dihitung sekali; konflik tim pada satu frame dikecualikan
dan dilaporkan. Tidak ada konflik pada kedua game ini.

Anchor umpan tidak menghukum fragmentasi di antara event. Hasil ini tidak memilih
pemenang spellF1 dan tidak mengukur akurasi lengkap H/A/D.

Lima kandidat aturan ditambahkan setelah implementasi dan tes sintetisnya selesai, kemudian
dibekukan sebelum benchmark dijalankan. Hasilnya belum mengalahkan baseline pada anchor ini;
tidak ada threshold yang diubah sesudah melihat angka tersebut. Kandidat dipertahankan sebagai
ablation, bukan dipromosikan ke default. Kedua sample tidak memiliki Smart Tagging, sehingga
hasil bola keluar dan +restart identik dan tidak mengukur manfaat restart bertag.

Supervised v1/v2 dilatih pada dataset IDSSE yang berbeda, bukan Metrica atau BAR-ATH.
Versi kedua meningkatkan ketahanan noise pada IDSSE tetapi anchor Metrica turun dari HMM
v2 menjadi 94,61%/94,71%. Hasil ini dicatat sebagai batas transfer lintas provider;
default tidak diganti. Evaluasi H/A/D lengkap eksternal kini tersedia terpisah di
[SUPERVISED_IMPROVEMENT.md](SUPERVISED_IMPROVEMENT.md), termasuk pengungkapan penggunaan
ulang game test sesudah audit v1.

Iterasi interaksi/handover meningkatkan spellF1 lengkap IDSSE tetapi menurunkan anchor
Metrica lagi. Joint salah tim pada 52/72 anchor dan salah memberi D pada 6/7 anchor, dibanding
baseline; angka ini tetap berupa evaluasi anchor parsial. Pergeseran kontrol H/A adalah masalah transfer
utama pada ukuran ini. Rincian post-hoc ada di `outputs/external_benchmark/error_categories.json`;
tidak ada parameter yang diubah berdasarkan diagnostik tersebut. Detail iterasi dan batas
evaluasi: [INTERACTION_IMPROVEMENT.md](INTERACTION_IMPROVEMENT.md).

## Reproduksi

```powershell
# Pastikan checkpoint terlatih sudah dibuat.
python tools/prepare_idsse.py
python tools/train_supervised.py --config config/exp_supervised_v1.toml
python tools/train_supervised.py --config config/exp_supervised_v2.toml --out-dir outputs/supervised_v2
python tools/train_supervised.py --config config/exp_supervised_v3.toml --out-dir outputs/supervised_v3
python tools/train_supervised.py --config config/exp_supervised_v4.toml --out-dir outputs/supervised_v4
python tools/train_boundaries.py
python tools/train_supervised.py --config config/exp_supervised_balanced.toml --out-dir outputs/supervised_balanced
python tools/select_joint.py
python tools/benchmark_external.py --download
```

Di Windows bisa memakai `benchmark_external.cmd`. Opsi `--download` mengambil hanya berkas
publik dari commit yang dipin. Salinan lokal yang hash-nya berbeda ditolak, bukan ditimpa diam-diam.
Tidak ada materi pertandingan kompetisi yang diunggah.

Laporan lengkap: `outputs/external_benchmark/report.md` dan `report.json`. CSV prediksi
sample game juga disimpan di folder tersebut, terpisah dari submission BAR-ATH. Metadata
mencatat hash sumber, config, checkpoint, kode dan hasil, serta menyatakan batas cakupan acuan.
Plan terbaru mengecualikan kandidat HMM terlatih yang telah dikeluarkan.

## Pengujian dan kesiapan CI

238 tes lokal lulus pada pipeline joint. Tes adapter memakai CSV sintetis untuk menguji unit koordinat, reset
kecepatan/babak, kesesuaian timeline, koordinat bola duplikat, deduplikasi/konflik anchor,
kesalahan offset, penghitungan D/X sebagai miss, isolasi event dari inferensi, dan verifikasi
hash unduhan sebelum menyimpan berkas final.

Workflow CI kini memiliki matrix Windows/Linux dan job agregat bernama `checks`, sehingga nama
status yang diwajibkan ruleset branch tetap tersedia. Sintaks dan kecocokan ruleset diperiksa
lokal. Status workflow jarak jauh dapat diperiksa pada PR terkait. CI memakai data sintetis;
dataset kompetisi maupun sample eksternal tidak diunduh dalam workflow.

Pemeriksaan artefak lokal satu perintah tersedia di [RELEASE_CHECK.md](RELEASE_CHECK.md).

## Pemeriksaan sumber tambahan

Sample Game 3 juga diperiksa pada commit sumber yang sama, melalui
[metadata EPTS](https://github.com/metrica-sports/sample-data/blob/e706dd506b360d69d9d123d5b8026e7294b13996/data/Sample_Game_3/Sample_Game_3_metadata.xml)
dan [event JSON](https://github.com/metrica-sports/sample-data/blob/e706dd506b360d69d9d123d5b8026e7294b13996/data/Sample_Game_3/Sample_Game_3_events.json).
Definisi channel tracking hanya menyediakan koordinat x/y. Sebanyak 3.620 event memiliki
tipe, tim, waktu/frame awal-akhir dan atribut pelaku, tetapi bukan label H/A/D setiap frame.
Karena itu belum ditemukan acuan lengkap bola mati pada sample ini. Interval antarevent
tidak diisi sebagai ground truth, dan Game 3 belum dimasukkan ke benchmark atau training.

Pekerjaan berikutnya: konfirmasi dengan game/provider baru yang memiliki acuan possession/bola
mati lengkap atau evaluasi panitia; eksekusi CI lintas platform dan review PR. Integrasi Tech 2/3 lokal telah
selesai dan dicatat di [README tim](../README.md). Tidak diperlukan anotasi
manusia BAR-ATH dan belum ada model final yang dipilih.
