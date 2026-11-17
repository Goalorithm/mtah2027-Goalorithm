# Interaksi pemain-bola dan model pergantian tim

13 November 2026. Target pengguna adalah spellF1 minimal 0,8. Target itu belum tercapai pada
hasil yang tersedia. Kenaikan relatif dan kelulusan tes kode tidak menggantikan target tersebut.
Dokumen ini melanjutkan [SUPERVISED_IMPROVEMENT.md](SUPERVISED_IMPROVEMENT.md).

Iterasi berikutnya memakai TCN lokal pada GPU dan meningkatkan rerata test asli menjadi
0,7950. Hasil training, seluruh profil dan batas target ada di
[TEMPORAL_IMPROVEMENT.md](TEMPORAL_IMPROVEMENT.md). Tabel di bawah mencatat tahap joint sebelumnya.

Ukuran utama tetap `spell_f1_mean` dari scorer panitia: rerata F1 pencocokan interval H/A
pada IoU 0,3/0,4/0,5/0,6/0,7. Skor frame atau ketepatan anchor yang lebih tinggi tidak
menggantikan metrik ini. Satu interval singkat yang terlewat tetap dihitung sebagai objek
yang gagal dideteksi, meskipun hanya sedikit frame yang salah.

## Data dan pemisahan

Training tetap memakai IDSSE J03WOH/J03WOY/J03WPY; validasi J03WQQ/J03WR9; evaluasi
J03WMX/J03WN1. Dua game evaluasi sudah terekspos sejak v1/v2 dan audit error. Hasil berikut
merupakan pengujian ulang, bukan holdout baru. Sumber/lisensi CC BY 4.0, pin file dan batas
semantik provider mengikuti dokumen sebelumnya. Scorer panitia tidak diubah. Label BAR-ATH,
contoh publik dan pseudo-label model sendiri tidak dipakai untuk fitting/pemilihan kandidat.

## Implementasi

- `interaction.py` menambah jarak pemain kedua, gerak relatif, keselarasan kecepatan,
  kedekatan beberapa pemain, minimum jarak lokal, umur kandidat kontak sebelum/sesudah frame,
  serta jarak tim pada offset waktu. Radius 0,75/1,5/2,5 m membentuk fitur; bukan label kontak
  yang dianggap pasti. Riwayat tidak melintasi batas babak.
- V3 memiliki 93 fitur, termasuk jendela pendek +/-5 dan +/-15 frame. Input tidak mencakup
  identitas, BallStatus/BallPossession, event, ball_extra atau citra. Training memakai bobot
  interval pendek/transisi, 220 round, 31 leaf/tree dan 511.164 baris sampling/augmentasi.
- V4 menguji 360 round/63 leaf. Varian possession-balanced memasukkan semua frame interval
  <=2 detik dan menyamakan bobot total setiap possession training; jumlah barisnya 604.452.
  Kedua varian ini tidak otomatis mengungguli v3. Hasil negatif dipertahankan.
- `boundary.py` dan `train_boundaries.py` mempelajari pergantian ke H, pergantian ke A,
  atau tidak ada pergantian. **D dalam classifier kejadian berarti tidak ada pergantian**,
  terpisah dari D bola mati pada classifier keadaan. Target berasal hanya dari game training,
  radius dua frame. Model kejadian memakai 160 round/31 leaf dan 511.164 baris.
- Decoder memakai keluaran model kejadian sebagai potensial transisi H/A. Bobot nol identik
  dengan Viterbi awal; transisi terlarang tetap terlarang. Bobot besar menghasilkan banyak
  pergantian palsu dan ditolak oleh validasi.
- `select_joint.py` memilih komponen yang sudah dibekukan pada validasi, tanpa fitting baru.
  Model/config/laporan komponen disalin ke `frozen_parent/` dan `frozen_events/`, dengan hash
  yang diperiksa kembali oleh pemeriksa rilis.

## Hasil validasi dan audit durasi

Rerata validasi mencakup dua game x tiga profil: asli, kehilangan bola dan kehilangan+jitter.

| Kandidat | Rerata spellF1 validasi |
| --- | ---: |
| Supervised v2 | 0,4986 |
| V3 interaksi | 0,6448 |
| V4 kapasitas lebih besar | 0,6394 |
| Possession-balanced | 0,6438 |
| V3 + handover, bobot 1 | 0,6657 |
| Possession-balanced + handover, bobot 0,5 | 0,6649 |

Pilihan deterministik adalah v3, skala emisi 0,3 dan bobot handover 1. Selisih 0,0009 terhadap
alternatif terdekat tidak membuktikan keunggulan statistik. Pada input asli validasi, kombinasi
terpilih mendapat 0,6988/0,7345. Ini bukan skor pertandingan kompetisi. Semua alternatif ada di
`outputs/supervised_joint/training_report.json`.

Sebelum penambahan handover, input asli v3 pada dua game evaluasi mendapat 0,6411/0,6288,
dibanding v2 0,4947/0,4806. Recall pada IoU 0,5:

| Durasi | J03WMX v2 -> v3 | J03WN1 v2 -> v3 |
| --- | ---: | ---: |
| <2 detik | 13,6% -> 26,7% | 15,2% -> 21,6% |
| 2-5 detik | 37,2% -> 51,8% | 33,1% -> 51,9% |
| >=5 detik | 72,1% -> 86,3% | 72,2% -> 85,2% |

Banyak interval pendek masih terlewat. Jumlah possession pendek yang sedikit tidak dipakai
sebagai ukuran keberhasilan. Audit: `outputs/idsse_benchmark_v3/duration_audit.json`.
Benchmark kombinasi lengkap: `outputs/idsse_benchmark_joint/report.json`.

## Hasil kombinasi terpilih pada game evaluasi

| Profil | V2 sebelumnya, rerata dua game | Joint, rerata dua game |
| --- | ---: | ---: |
| Asli | 0,4876 | 0,7202 |
| Bola hilang sintetis | 0,4331 | 0,6043 |
| Bola hilang + jitter 0,15 m | 0,4267 | 0,5943 |

Pada input asli, joint mendapat 0,7233/0,7172. Interval bootstrap 5-95% per game adalah
0,6909-0,7578 dan 0,6856-0,7513. Ini interval kondisional pada game/model tersebut, bukan
ketidakpastian lintas provider atau akibat pengembangan berulang pada game terekspos.
Target 0,8 belum dicapai. Pada IoU 0,5, recall interval <2 detik naik menjadi 41,9%/46,2%,
interval 2-5 detik 69,3%/67,5%, dan >=5 detik 91,8%/89,9%.

**Transfer provider masih lemah:** pada Metrica PASS-start, joint mendapat 92,73%/91,80%,
di bawah HMM lama 96,24%/97,93% dan baseline 93,86%/96,27%. Anchor ini mengukur metrik berbeda,
tetapi penurunannya tetap merupakan bukti bahwa keberhasilan IDSSE belum universal.

CSV BAR-ATH joint berisi 199.319 frame, 458 possession (58 <2 detik), D 34,00% dan X 0% pada
frame yang dinilai. Statistik ini tidak mengukur akurasi BAR-ATH. SHA-256:
`2d063850f6b6be84780333f68000205a4677363ba85d30495b8298f445c25ab2`.

Pemeriksaan utama lulus dengan 238 tes dan sembilan pemeriksaan lokal. Seluruh 18 varian
memiliki dua pengulangan identik. Benchmark mencakup 42 evaluasi H/A/D lengkap dan 32 evaluasi
anchor (16 konfigurasi x dua game). Model default dan keputusan submission belum diganti.
Riset tambahan sudah menguji campuran forest dan sampling yang mempertahankan semua frame
transisi positif. Rerata validasi masing-masing 0,6692 dan 0,6682, dibanding joint 0,6657.
Keduanya tidak memenuhi ambang kenaikan 0,01 yang dicatat sebelum hasil riset lengkap,
sehingga tidak diprioritaskan. Evaluasi ulang input asli juga sekitar 0,72. Kalibrasi peluang
handover memilih bias nol pada validasi: tidak ada perubahan numerik yang diadopsi.
Laporan lengkap: `outputs/idsse_benchmark_research/report.json`,
`outputs/v3_workflow/research_priority_decision.json`, dan
`outputs/supervised_calibrated/zero_bias_identity.json`.

## Pelatihan akhir setelah pemilihan parameter

`tools/refit_external.py` melatih ulang classifier keadaan dan handover pada kelima game
train+validasi, setelah seluruh parameter dipilih dari pemisahan 3/2 sebelumnya. Jumlah data
menjadi 715.116 frame berlabel asli atau 858.162 baris sampling/augmentasi. Dua game evaluasi
tetap dikecualikan dari fitting. Skala emisi 0,3 dan bobot handover 1 tidak dipilih ulang.
Laporan refit tidak mengklaim skor validasi baru: game validasi lama kini dipakai untuk fitting.
Skor pemilihan induk ditandai eksplisit sebagai skor sebelum refit.

`run_refit.cmd` mereproduksi tahap ini, CSV penuh, dua pengulangan eksperimen, evaluasi H/A/D,
anchor Metrica dan pemeriksaan artefak. Hasil refit harus dibaca bersama pengujian transfer,
bukan diasumsikan lebih baik karena jumlah data lebih besar. CSV/parameter/laporan tetap
disimpan lokal di direktori yang di-ignore Git.

Evaluasi refit telah selesai: 12 hasil (dua model x tiga profil x dua game), tanpa pemilihan
parameter tambahan. Rerata spellF1:

| Profil | Joint sebelum refit | Refit lima game |
| --- | ---: | ---: |
| Asli | 0,7202 | 0,7129 |
| Bola hilang sintetis | 0,6043 | 0,6004 |
| Bola hilang + jitter | 0,5943 | 0,5931 |

Pada anchor Metrica, refit mendapat 731/798 (91,60%) dan 885/964 (91,80%). Tambahan data dari
provider yang sama belum memberi peningkatan. Joint tetap acuan utama dokumen; refit adalah
hasil eksperimen tambahan, bukan keputusan submission. Menggunakan hasil test terekspos
untuk merancang iterasi berikutnya tidak akan menghasilkan holdout independen.

Laporan: `outputs/idsse_benchmark_refit/report.json`, `duration_audit.json`, dan
`outputs/external_benchmark_refit/report.json`. Kedua pengulangan CSV BAR-ATH refit identik:
199.319 baris, 458 possession/58 <2 detik, D 34,39%, X 0% pada frame dinilai. Ini diagnostik
struktur, bukan akurasi BAR-ATH. SHA-256:
`63f560723025fbc675988da6b78829d0345307d2176c3931323f0c6da22d6dec`.

Pemeriksaan refit terpisah juga lulus sembilan pemeriksaan lokal dan 238 tes;
`outputs/release_check_refit/report.json` tetap menandai target 0,8 belum terpenuhi pada
semua profil, baik rerata maupun setiap game. Tidak ada model yang dikirim atau dipublikasikan.

## Reproduksi dan pemeriksaan

`run_all_tech.cmd` mencakup seluruh training, pemilihan, inferensi, benchmark dan pemeriksaan.
Setelah komponen v3/boundary/balanced tersedia, tahap pemilihan dapat diulang:

```powershell
python tools/select_joint.py
python -m mtah.predict --config config/exp_supervised_joint.toml --name exp_supervised_joint --no-score
python tools/benchmark_idsse.py --config config/exp_supervised_joint.toml --plan config/idsse_benchmark_joint.toml --out-dir outputs/idsse_benchmark_joint
python tools/audit_spells.py --benchmark outputs/idsse_benchmark_joint/report.json --out outputs/idsse_benchmark_joint/duration_audit.json
python tools/check_release.py
```

JSON pohon memakai angka finite, termasuk rute eksplisit untuk percabangan khusus nilai hilang.
Predictor CPU scikit-learn 1.7.2 dan NumPy memberi probabilitas identik dalam pengujian; CSV v2/v3
pada game nyata juga cocok setelah pergantian backend. Evaluasi batch memakai ulang fitur dan
keluaran numerik yang sama, lalu menjalankan decoder setiap kandidat sendiri. Dua pengulangan
eksperimen memakai cache terpisah. Seluruh 238 tes lokal lulus pada perubahan kode ini.

## Data berikutnya

Metadata [SkillCorner open data](https://github.com/SkillCorner/opendata) dipin ke commit
`4340d274572876239c154c90bc507a9b3250a656`. README, LICENSE dan daftar match disimpan lokal;
audit ada di `outputs/external_research/skillcorner_audit.json`. Provider menjelaskan tracking
broadcast 10 fps, possession dan fase yang diturunkan oleh Game Intelligence. Metadata ini belum
dipakai untuk training atau sebagai ground truth. Kelengkapan fase dan arti possession kosong
perlu ditetapkan sebelum membuat acuan D; area kosong tidak otomatis diberi D.

Langkah berikutnya adalah memperkuat deteksi kontrol singkat dengan data eksternal lebih beragam
dan mengonfirmasi transfer pada provider/game yang belum dipakai. Target 0,8 memerlukan bukti
baru; pemilihan ulang pada dua game terekspos tidak membuktikannya.
