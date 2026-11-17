# Peningkatan supervised dan ketahanan noise

Catatan versi 11 November. Peningkatan 13 November dan hasil joint terbaru tersedia di
[INTERACTION_IMPROVEMENT.md](INTERACTION_IMPROVEMENT.md).

11 November 2026. Kandidat baru memakai label **game eksternal**, bukan label manusia BAR-ATH.
Peningkatan utama adalah evaluasi interval possession/H/A/D lengkap, supervised training, dan
perbaikan kelemahan noise yang ditemukan pada versi pertama. Config default tetap baseline.

## Data, asal label, dan pemisahan game

Sumber: Bassek, Rein, Weber dan Memmert, *An integrated dataset of spatiotemporal and event data
in elite soccer*, Scientific Data 12, 195 (2025),
[doi:10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y).
Data resmi DFL/Sportec tersedia dengan lisensi [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
[DOI dataset](https://doi.org/10.6084/m9.figshare.28196177).

Unduhan memakai [mirror PySport](https://huggingface.co/datasets/pysport/idsse-data), revision
`a715a38dfbaf5f58e431727c2b78d174101a703c`. Lima belas file (tujuh metadata, tujuh tracking,
satu README) dipin berdasarkan ukuran dan SHA-256 di `config/idsse_sources.json`.
Total tracking sekitar 2,62 GB. Metadata/XML aktual menentukan ID game; daftar ID dalam README
mirror tidak seluruhnya mutakhir. Dataset tidak berisi video dan tidak digabung dengan event
BAR-ATH dari pihak lain.

| Peran | Game | Frame berlabel asli |
| --- | --- | ---: |
| Training | J03WOH, J03WOY, J03WPY | 425.961 |
| Validasi/pemilihan decoder | J03WQQ, J03WR9 | 289.155 |
| Pengujian di luar training/validasi | J03WMX, J03WN1 | 287.528 |

Provider menyediakan `BallStatus` (0 inactive, 1 active) dan `BallPossession` (1 home, 2 away).
Adapter mengubah inactive menjadi D dan active+owner menjadi H/A. Owner tidak dikenal saat
active tetap X dan dikecualikan dari acuan, tidak diisi dengan tebakan. Tidak ada anotasi atau
koreksi label baru oleh tim ini. Proses pengumpulan dan pengendalian mutu provider dijelaskan
oleh paper; detail anotasi setiap channel tidak diasumsikan melebihi informasi sumber.

BallStatus/Possession hanya keluar sebagai array referensi terpisah. Tracking untuk model
tidak memuat keduanya, tidak memakai Z/ball_extra, event, ID pemain, nama klub, atau waktu absolut.
Koordinat centered-metre dipetakan ke konvensi 105,3 x 68 m, S pemain diubah km/h ke m/s, dan
timestamp pemain harus cocok dengan clock bola 25 fps. Satu separator X memisahkan kedua babak
untuk scorer; tidak ada smoothing, interpolasi atau state yang menyeberang halftime.

Saat pembacaan test pertama ditemukan track `TeamId=referee`. Track tersebut dikecualikan
secara eksplisit; pemain yang tidak cocok dengan metadata tetap ditolak. Perbaikan format ini
dilakukan sebelum skor test pertama berhasil dihitung. Pengulangan training membuktikan
parameter dan decoder v1 tetap identik. Auditnya tersimpan di `outputs/idsse_benchmark/ingestion_audit.json`.

## Model dan training

Histogram Gradient Boosting mempelajari H/A/D dari fitur tracking, lalu Viterbi mencari urutan
keadaan dengan transisi/initial/prior yang dipelajari hanya dari referensi tiga game training.
H/A disimetriskan melalui pertukaran fitur/label saat training dan rata-rata probabilitas
kedua orientasi saat inferensi. Identitas klub tidak menjadi fitur.

V1 memakai 26 fitur: jarak dua tim, bukti tim sebelum/sesudah kehilangan bola, umur observasi,
jarak ke garis, kecepatan bola/pemain, jumlah objek terlacak, dan jendela bukti +/-1 dan +/-3 detik.
V2 menambah tiga fitur gerak dari perpindahan bersih selama lima frame. Ini mengurangi respons
terhadap jitter per-frame sambil mempertahankan fitur lama v1.

Model v2 dilatih pada **511.164 baris fitur setelah sampling/augmentasi**, bukan 511.164 frame
unik baru. Stride 5, tiga profil (asli, bola hilang, bola hilang+jitter), dan pertukaran H/A
menghasilkan baris tersebut dari tiga game training. Profil kehilangan bola memakai proses
Markov dengan fraksi 0,4 dan rata-rata durasi 25 frame. Jitter Gaussian 0,15 m diterapkan tanpa
melihat label dan tanpa mengisi bola yang hilang. Noise ini adalah skenario uji, bukan estimasi
bahwa noise kompetisi persis 0,15 m atau independen antar-frame.

Parameter lain: 120 iterasi, maksimal 15 leaf/tree, learning rate 0,08, L2 5, minimum 50 sampel
per leaf, seed 20261111 dan satu thread numerik. Model memiliki 360 tree dan 29 fitur.
Tidak ada refit memakai game test. Skala decoder dipilih dari 0,03/0,1/0,3/1 berdasarkan rerata
spellF1 validasi: dua game x tiga profil. Skala 1 terpilih dengan rerata 0,4986; skala 0,3
mendapat 0,4904. Selisih kecil ini bukan klaim bahwa skala 1 selalu unggul pada game baru.

Checkpoint berupa parameter tree JSON, bukan pickle. Inferensi NumPy diuji terhadap
`scikit-learn==1.7.2`, termasuk cabang NaN; selisih probabilitas pada pemeriksaan training adalah
0. Parameter, laporan dan receipt dari dua training v2 independen identik byte-per-byte.

## Audit v1 dan batas independensi evaluasi

V1 pertama dibekukan sebelum membaca referensi dua game test. Hasil tracking asli meningkat
dari rerata spellF1 HMM lama 0,3336 menjadi 0,4882. Namun, audit game training menemukan bahwa
koordinat bola **tetap persis pada 99,998% frame D**. Kecepatan pemain yang diekspor tetap
masuk akal dibanding turunan posisi, sehingga tidak ditemukan indikasi bahwa kolom itu
menyalin label. Konvensi koordinat bola beku tetap membuat deteksi D jauh lebih mudah.

Stress test tambahan setelah hasil awal, dengan kehilangan bola+jitter independen 0,15 m,
menjatuhkan v1 ke sekitar spellF1 0,25. Hasil tersebut dipertahankan dan tidak digunakan untuk
menyatakan kandidat tahan noise. V2 menambah augmentasi jitter dan fitur net-motion.

**Dua game test v2 sudah pernah dipakai untuk evaluasi/audit v1.** Angka v2 di bawah adalah
pengujian ulang pada game di luar training/validasi, bukan holdout baru yang belum pernah dilihat.
Parameter fit dan pemilihan decoder v2 tetap hanya menggunakan tiga game training dan dua
validasi. Pengembangan v2 dipicu audit robustness v1; fakta tersebut dicatat dalam config,
laporan training dan benchmark. Tidak ada tuning pada label publik BAR-ATH atau pseudo-label sendiri.

## Hasil pengujian ulang v2

Scorer memakai fungsi panitia yang sama: spellF1 rata-rata IoU 0,3–0,7, macroF1 dan D F1.
Semua model pada satu baris menerima tracking/mask/noise yang sama. Tidak ada Smart Tagging
eksternal yang diberikan ke model mana pun.

| Profil | Game | HMM lama spellF1 | Supervised v1 | Supervised v2 |
| --- | --- | ---: | ---: | ---: |
| Asli | J03WMX | 0,3587 | 0,4977 | 0,4947 |
| Asli | J03WN1 | 0,3086 | 0,4787 | 0,4806 |
| Bola hilang | J03WMX | 0,2898 | 0,4434 | 0,4387 |
| Bola hilang | J03WN1 | 0,2474 | 0,4283 | 0,4275 |
| Bola hilang+jitter | J03WMX | 0,2627 | 0,2560 | 0,4270 |
| Bola hilang+jitter | J03WN1 | 0,2144 | 0,2275 | 0,4263 |

V2 mempertahankan sebagian besar kinerja bersih/occlusion v1 sambil memperbaiki keruntuhan
pada noise. Ia tidak unggul atas v1 pada setiap baris. Pada profil tersulit, macroF1 v2
0,8797/0,8447 dibanding HMM 0,7091/0,6315. Bootstrap resmi 500 draw per baris memberi interval
5–95% spellF1 v2 0,3886–0,4650 dan 0,3923–0,4647; interval HMM 0,2364–0,2911 dan 0,1911–0,2384.
Interval ini tidak mengukur ketidakpastian lintas liga, provider, atau pemilihan model.

Laporan lengkap: `outputs/idsse_benchmark_v2/report.json` dan `report.md` (24 evaluasi).
Hasil v1 awal/stress tetap tersedia di `outputs/idsse_v1_initial/` dan `outputs/idsse_stress/`.

## BAR-ATH dan reproduksi

Prediksi baru di `outputs/exp_supervised_v2.csv`: 199.319 baris, 474 possession, 101 di bawah
dua detik, D 37,11% dan X 0% pada frame yang dinilai. HMM lama menghasilkan 313/32 pendek.
**Perubahan struktur tersebut bukan bukti bahwa v2 lebih akurat pada BAR-ATH.** Skor penuh
pertandingan kompetisi belum tersedia dan tidak ada model pemenang yang dipilih otomatis.

SHA-256 CSV v2: `0d18868a6c3887e59453a0c45add119fbdd675f1ffbb2d1ea828af3ce2e5b47e`.

Pemeriksaan transfer pada anchor Metrica juga dijalankan. V2 mendapat 94,61%/94,71%, di bawah
HMM lama 96,24%/97,93%. Ini kelemahan lintas provider yang tetap dicatat; peningkatan pada
spellF1 IDSSE belum membuktikan keunggulan universal atau menjadi alasan mengganti default.

Perintah benchmark di bawah mengasumsikan comparator HMM/v1 sudah tersedia. Pada laptop baru,
gunakan `run_all_tech.cmd` untuk menyiapkan seluruh comparator dan pemeriksaan.

```powershell
python tools/prepare_idsse.py
python tools/train_supervised.py --config config/exp_supervised_v2.toml --out-dir outputs/supervised_v2
python -m mtah.predict --config config/exp_supervised_v2.toml --name exp_supervised_v2 --no-score
python tools/benchmark_idsse.py --config config/exp_supervised_v2.toml --plan config/idsse_benchmark_v2.toml --out-dir outputs/idsse_benchmark_v2
```

`run_all_tech.cmd` juga menyiapkan v1/HMM, eksperimen, benchmark anchor dan pemeriksaan rilis.
Semua data, cache, checkpoint, prediksi dan laporan tetap lokal/ignored. Data/video kompetisi
tidak diunggah. Default baseline, kode panitia dan edit pengguna pada RULES.md dipertahankan.

Langkah paling bernilai berikutnya: evaluasi pada game berlabel dari provider lain yang belum
pernah dipakai, uji gangguan tracking yang lebih beragam, reproduksi lintas mesin, review tim,
dan penilaian panitia. Dataset corrected-tracking ini memperkuat bukti, tetapi tidak menjamin
transfer ke seluruh kegagalan tracking otomatis maupun hasil juara.

## Audit terhadap target pengguna spellF1 0,8

Pengguna menetapkan 0,8 sebagai target minimum. Rerata 0,4876 belum memenuhi target itu;
kenaikan relatif terhadap HMM tidak mengubah status tersebut. Tes kode yang lulus juga bukan
bukti terpenuhinya target akurasi. Audit berikut bersifat post-hoc pada dua game yang sudah
terekspos, tanpa fitting, perubahan decoder, atau pemilihan model berdasarkan label test.

| Input asli v2 | J03WMX | J03WN1 |
| --- | ---: | ---: |
| macroF1 | 0,9033 | 0,8784 |
| D F1 | 0,9992 | 0,9983 |
| spellF1 | 0,4947 | 0,4806 |
| Possession prediksi / acuan | 321 / 511 | 329 / 500 |
| Recall possession <2 detik pada IoU 0,5 | 26/191 (13,6%) | 26/171 (15,2%) |
| Recall possession >=5 detik pada IoU 0,5 | 132/183 (72,1%) | 122/169 (72,2%) |

Dengan jumlah possession prediksi tetap, batas atas F1 pencocokan satu-ke-satu adalah
`2 * min(n_pred, n_true) / (n_pred + n_true)`: 0,7716 dan 0,7937. Batas ini bahkan mengandaikan
seluruh interval prediksi cocok sempurna; model saat ini tidak bisa mencapai 0,8 hanya dengan
menggeser batas waktu sambil mempertahankan jumlah interval. Jumlah acuan ini hanya dipakai
untuk diagnosis eksternal, bukan target jumlah yang dipaksakan pada BAR-ATH.

Eksperimen oracle diagnostik mengganti label tim dengan acuan hanya ketika prediksi dan acuan
sama-sama hidup, sambil mempertahankan kesalahan deteksi D yang tersisa. SpellF1 menjadi
0,9934/0,9906. Ini **bukan skor model** dan tidak dapat digunakan untuk submission; penggantian
memakai label acuan secara langsung. Hasil ini mengisolasi kesalahan H/A sebagai hambatan utama
pada input asli provider ini. Deteksi D yang hampir sempurna tidak menutupi possession pendek
yang hilang atau batas pergantian tim yang keliru.

Prioritas pengembangan berikutnya adalah fitur kontrol/interaksi pemain-bola dan detektor
pergantian tim yang mempertahankan interval pendek yang sah, lalu decoder temporal yang
dipilih pada game training/validasi. Mengurangi smoothing saja berisiko menambah false positive.
Pencapaian target harus dikonfirmasi pada game/provider baru yang belum digunakan setelah
revisi arsitektur. Tidak ada bukti saat ini yang menjamin target 0,8 akan tercapai.

Audit numerik: `outputs/idsse_benchmark_v2/error_audit_target_080.json`.
