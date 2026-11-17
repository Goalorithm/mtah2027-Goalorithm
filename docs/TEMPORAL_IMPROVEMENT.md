# Eksperimen TCN lokal pada GPU

13 November 2026. Kelanjutan [INTERACTION_IMPROVEMENT.md](INTERACTION_IMPROVEMENT.md).
Target tetap spellF1 0,8 dari scorer panitia; skor frame tidak menggantikannya.

Eksperimen berikutnya tersedia di [DOMAIN_IMPROVEMENT.md](DOMAIN_IMPROVEMENT.md): model batas
interval, label H/A parsial broadcast dan refit akhir. Tabel dokumen ini tetap hasil TCN awal.

## Metode dan batas data

Model baru memakai residual temporal convolution dengan dua head: keadaan H/A/D dan kejadian
pergantian ke H/ke A/tanpa pergantian. Head kejadian memakai kelas 2 untuk **tanpa pergantian**,
sedangkan kelas 2 pada head keadaan berarti **bola mati**. Keduanya tidak dipertukarkan.

Masukan neural adalah 93 fitur tracking yang sama, ditambah indikator nilai hilang. Konvolusi
memakai 64 channel dan dilation 1/2/4/8/16/32, dua konvolusi kernel 3 setiap blok: 160.582
parameter, radius 126 frame, konteks konvolusi 253 frame. Ini model noncausal untuk analisis
offline; konteks sebelum/sesudah frame diizinkan. Semua konteks diputus pada batas babak.
Inferensi per potongan memakai overlap radius tepat dan dibandingkan dengan urutan penuh.

Arsitektur terinspirasi studi [temporal convolution Bai dkk.](https://arxiv.org/abs/1803.01271),
dengan implementasi noncausal dan dua head khusus proyek ini. Runtime opsional dipin ke
PyTorch 2.9.1, memakai wheel CUDA 12.8 dari [indeks resmi](https://pytorch.org/get-started/previous-versions/).
Training berlangsung pada RTX 4070 Laptop lokal, tanpa mengunggah data/video kompetisi.

- Fitting: J03WOH/J03WOY/J03WPY; validasi: J03WQQ/J03WR9.
- Evaluasi: J03WMX/J03WN1; keduanya sudah terekspos pada iterasi sebelumnya. Ini **bukan**
  holdout baru dan tidak membuktikan generalisasi pada pertandingan yang belum dilihat.
- Normalisasi hanya dihitung pada training, dengan statistik H/A diikat simetris.
- Input tidak memasukkan reference BallStatus/Possession, event, identitas pemain, ball Z,
  label BAR-ATH atau pseudo-label sebagai acuan training.
- Augmentasi memakai kehilangan bola/noise yang tidak bergantung pada label. Sampling dan
  bobot loss pada training menekankan interval singkat serta area transisi.
- Optimizer AdamW, 35 epoch, 220 step/epoch, batch 16, core 256 frame. Training CUDA memakai
  bfloat16; inferensi memakai float32. Tidak ada klaim bit-identik lintas GPU/OS.

Setiap lima epoch, checkpoint dinilai pada enam profil validasi dengan campuran keadaan
neural 0,5 dan head kejadian induk. Setelah epoch dipilih, campuran keadaan
0/0,25/0,5/0,75/1, campuran kejadian 0/0,5 dan skala emisi 0,3/1 dipilih pada validasi.
Komponen pohon joint tetap dibekukan; bobot neural nol memungkinkan hasil negatif dicatat.
Skor test tidak dipakai untuk memilih epoch atau campuran. Dataset dan dua game evaluasi
tetap sama, sehingga penambahan kompleksitas tidak otomatis menjadi bukti peningkatan.

## Reproduksi

Setelah `run_all_tech.cmd` menyediakan induk joint dan data, pasang dependensi opsional:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-temporal.txt --index-url https://download.pytorch.org/whl/cu128
```

Jalankan `run_temporal.cmd`. Alur terpisah:

```powershell
python tools/train_temporal.py
python tools/evaluate_temporal.py --mode test
python tools/evaluate_temporal.py --mode anchors
python tools/evaluate_temporal.py --mode competition
```

`outputs/temporal/` menyimpan bobot, normalisasi, riwayat training, seluruh pemilihan validasi,
snapshot induk, receipt pembekuan dan laporan evaluasi. Bobot dimuat dengan `weights_only=True`.
Hash config, model, sumber, helper dan induk diverifikasi sebelum evaluasi. CSV penuh berada
di `outputs/exp_temporal.csv`, divalidasi format dan diulang dua kali. Ini kandidat lokal,
bukan submission atau keputusan juara. Tidak ada perubahan pada inventaris `src`, baseline,
scorer panitia atau artefak model lama; metode neural tersedia melalui CLI terpisah.

Sembilan tes tambahan memeriksa kesetaraan inferensi per potongan, isolasi babak, simetri tim,
normalisasi/nilai hilang, pemisahan game, roundtrip bobot, serta normalisasi probabilitas float32 ke kontrak decoder float64. Tes ini memerlukan PyTorch;
lingkungan baseline tanpa dependensi opsional melewati modul tes neural tersebut.

## Training dan pemilihan yang sudah selesai

Seluruh 35 epoch sudah dijalankan. Loss training turun dari 0,49470 ke 0,09198; angka ini
bukan skor evaluasi. Epoch 25 dipilih oleh validasi, meskipun loss epoch berikutnya lebih kecil.
Campuran akhir memakai 75% probabilitas keadaan TCN dan 25% pohon, skala emisi 0,3.
Head kejadian TCN diberi bobot nol pada inferensi: validasi mempertahankan head handover
pohon induk. Head neural tetap dilatih sebagai tugas bantu; tidak diklaim dipakai oleh decoder.

| Game validasi | Asli | Bola hilang | Bola hilang + jitter |
| --- | ---: | ---: | ---: |
| J03WQQ | 0,7704 | 0,7233 | 0,6991 |
| J03WR9 | 0,8258 | 0,7330 | 0,6963 |

Rerata enam profil naik dari pembanding joint 0,66572 ke 0,74132. Rerata input asli adalah
0,79809: satu game melampaui 0,8, game lainnya belum. Ini **hasil validasi**, tidak menggantikan
skor test atau membuktikan pencapaian target universal. Seluruh 20 kombinasi akhir dicatat
di `outputs/temporal/training_report.json`; model dibekukan sebelum membaca game evaluasi
pada iterasi TCN ini. Dua game tersebut tetap pernah terekspos pada iterasi sebelumnya.

Sembilan pemeriksaan utama juga lulus dengan 238 tes lokal. Pemeriksaan utama masih memvalidasi
artefak joint lama; TCN memeriksa hash/freeze sendiri melalui `evaluate_temporal.py`,
serta validasi format dan pengulangan CSV pada mode competition.

## Hasil test dan transfer

| Kondisi | Joint sebelumnya | TCN + pohon |
| --- | ---: | ---: |
| Asli | 0,7202 | 0,7950 |
| Bola hilang sintetis | 0,6043 | 0,6905 |
| Bola hilang + jitter | 0,5943 | 0,6732 |

Pada input asli, J03WMX mendapat 0,781132 dan J03WN1 0,808870. Rerata tepatnya
0,7950010958, **masih di bawah 0,8**; pembulatan dua desimal tidak mengubah status target.
Interval bootstrap 5-95% per game adalah 0,7472-0,8118 dan 0,7788-0,8418. Interval ini
kondisional pada game/model tersebut; tidak mencakup ketidakpastian akibat penggunaan test
berulang, pemilihan model atau pergantian provider.

Recall interval pada IoU 0,5:

| Durasi | J03WMX, joint -> TCN | J03WN1, joint -> TCN |
| --- | ---: | ---: |
| <2 detik | 41,9% -> 52,9% | 46,2% -> 57,9% |
| 2-5 detik | 69,3% -> 77,4% | 67,5% -> 80,6% |
| >=5 detik | 91,8% -> 92,9% | 89,9% -> 92,3% |

Transfer Metrica naik sedikit ke 741/798 (92,86%) dan 889/964 (92,22%). HMM teratur tetap
lebih tinggi, 96,24%/97,93%. Kesalahan tim masih 57/74 anchor; false D turun menjadi 0/1.
Dengan demikian peningkatan IDSSE belum menyelesaikan generalisasi kontrol tim antarprovider.
Metrica mengukur anchor awal umpan, bukan spellF1 lengkap.

CSV BAR-ATH berisi 199.319 frame dan identik dalam dua pengulangan CUDA lokal: 474 possession,
45 <2 detik, D 27,49%, X 0% pada frame dinilai. Penurunan D dibanding joint 34,00% tidak
dianggap peningkatan akurasi; acuan penuh BAR-ATH belum tersedia. SHA-256:
`df30c1037edcd88f47c5161d6e4fb5a6a58c0351b100fa8ae3375bc277bf4d75`.

Laporan terpisah di `outputs/temporal/`: `test_report.json` (enam evaluasi, bootstrap dan audit
durasi), `anchors_report.json`, `competition_report.json`, `training_report.json` dan
`learning_curve.png`. Flag target rerata dan syarat semua game masih false pada seluruh profil.

`verification.json` memeriksa keenam CSV test beserta hash/model/receipt/sumber, kesamaan
skor pembanding dengan benchmark joint sebelumnya, format/mask babak CSV penuh dan kecocokan
salinannya di `evaluation/results/`. Seluruh skor, anchor dan hash prediksi juga berhasil
direproduksi setelah perbaikan penamaan output agar kandidat berbeda memakai nama CSV sendiri.

Pemeriksaan backend pada 4.096 frame validasi mencatat selisih probabilitas maksimum sekitar
0,000947 saat CuDNN memakai TF32, dan 0,000000626 saat TF32 dimatikan, dibanding CPU. Kedua
head mempunyai argmax yang sama pada potongan itu. Ini tidak membuktikan identitas CSV seluruh
pertandingan lintas backend; hasil dan mode presisi dicatat di `backend_parity.json`.

Prioritas berikutnya tetap deteksi kontrol singkat dan data eksternal lintas provider dengan
acuan lengkap. Uji pada game/provider yang belum dipakai diperlukan sebelum menyatakan target
0,8 stabil atau menjadikannya keputusan submission. Tidak ada jaminan juara.

## Riset kekuatan decoder

Sesudah TCN dibekukan, `tools/select_temporal_decoder.py` menguji 15 kombinasi kekuatan
handover dan skala emisi pada enam profil validasi, tanpa fitting classifier tambahan.
Bobot handover 0,25/skala 0,3 mendapat 0,74754, naik 0,00622 dari 0,74132. Ini di bawah
margin adopsi 0,01 yang dicatat ketika grid masih berjalan (sebagian hasil sudah terlihat).
Varian `outputs/temporal_decoder/` tidak diprioritaskan dan tidak dievaluasi pada test.
Kebijakan/keputusan ada di `outputs/temporal_workflow/decoder_adoption_policy.json` dan
`decoder_decision.json`. Model TCN utama dan hasil test di atas tetap sama.
