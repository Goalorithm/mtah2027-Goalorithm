# Model batas interval dan fine-tuning lintas provider

Kelanjutan [TEMPORAL_IMPROVEMENT.md](TEMPORAL_IMPROVEMENT.md), 13 November 2026.

> Catatan 17 November: dokumen ini mencatat penelitian pada tracking eksternal bersih.
> Alur bola video dan artefak turunannya telah dikeluarkan dari branch. Laporan historis
> perlu dibangun ulang setelah perubahan sumber sebelum dipakai memverifikasi checkpoint.
Target tetap `spell_f1_mean` 0,8. Metrik klasifikasi titik dan fungsi pemilihan internal tidak
menggantikan skor interval panitia.

## Eksperimen TCN yang lebih panjang

TCN v2 memiliki 406.950 parameter, 96 channel dan konteks konvolusi 509 frame (sekitar 20 detik).
Training 45 epoch menambahkan bobot penguasaan pendek 0,75, dropout fitur 5%, dropout kelompok
fitur bola 15%, noise fitur kecil dan loss konsistensi pergantian keadaan berbobot 0,1.
Loss terakhir memakai peluang perubahan antara dua posterior keadaan bersebelahan, mengabaikan
padding/label X. Ini implementasi sendiri, dengan referensi metodologis
[ASRF](https://arxiv.org/abs/2007.06866) dan
[MS-TCN](https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html),
bukan klaim reproduksi kedua arsitektur tersebut.

Validasi memilih epoch 30 dan campuran keadaan neural/pohon 50/50. Rerata validasi 0,74389
dibanding TCN pertama 0,74132; selisih kecil ini tidak membuktikan keunggulan statistik.
Ensemble kedua TCN mendapat 0,74670, tetapi gain 0,00281 terhadap komponen terbaik tidak memenuhi
margin adopsi 0,005 yang ditetapkan sebelum pemilihan ensemble. Ensemble ini tidak dievaluasi
pada test. Model dan laporan lama dipertahankan.

## Data broadcast tambahan dan batas label

Sumber: [SkillCorner Open Data](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/README.md),
commit `4340d274572876239c154c90bc507a9b3250a656`. Tracking berasal dari broadcast dan label
penguasaan/fase berasal dari Game Intelligence. Ini **silver labels algoritmik**, bukan
anotasi manusia yang telah diverifikasi. Empat file tracking sekitar 363 MB diunduh dari
Git LFS dengan SHA-256/ukuran yang diverifikasi; metadata match dan fase dipin pada commit sama.

| Peran | Game SkillCorner |
| --- | --- |
| Training tambahan | 1886347, 1899585 |
| Validasi tambahan | 1925299 |
| Test game terpisah | 1927964 |

Peran ditetapkan sebelum frame tracking/label game test diparsing. Metadata/schema match
telah diperiksa saat menyiapkan adapter; game ini tidak diklaim sebagai test yang seluruh
informasinya tersembunyi. Game 1874553 yang pernah diperiksa pada riset sebelumnya tidak
masuk pemisahan ini. Dua game test IDSSE tetap game yang telah digunakan berulang kali.

Acuan H/A hanya disimpan ketika tiga syarat bersamaan terpenuhi: bola terdeteksi, pemilik
tim diketahui, dan tim itu cocok dengan interior fase bermain aktif. Endpoint fase dikecualikan.
Kepemilikan kosong, bola hasil extrapolation, konflik fase dan ketiadaan fase semuanya tetap X.
**Tidak dibuat label D dari ketiadaan fase atau pemilik.** Akurasi D dan spellF1 H/A/D lengkap
tidak dapat diukur dengan acuan parsial ini.

Tracking 10 Hz dipetakan ke model 25 fps. Posisi pemain diinterpolasi hanya di antara dua
observasi sumber bersebelahan yang valid; bola dan label tetap jarang pada waktu sumber
terdekat, dengan kuantisasi maksimum 0,02 detik. Kecepatan dihitung ulang pada 25 fps.
Konversi koordinat memakai ukuran lapangan sumber; urutan, kecepatan dan interpolasi diputus
pada batas babak. Identitas, pemilik, fase dan ball Z tidak masuk fitur inferensi. Metadata
roster hanya menentukan slot dan tim pemain.

## Fine-tuning multi-provider

TCN v2 dipilih sebagai warm start berdasarkan validasi. Normalisasi dan hyperparameter decoder
50/50, skala 0,3 serta head handover pohon dibekukan. Seluruh parameter neural dapat diperbarui.
Fine-tuning berjalan 20 epoch dengan learning rate 0,0001. Pool crop training terdiri atas
75% crop IDSSE dan 25% SkillCorner; bobot label parsial diketahui dikalikan empat, label X
tidak menyumbang loss. Kejadian/batas yang tidak diketahui tidak mendapat target buatan.

Fungsi pemilihan epoch adalah rerata spellF1 validasi IDSSE ditambah 0,2 kali kenaikan akurasi
H/A parsial SkillCorner. Penurunan spellF1 IDSSE dibatasi maksimum 0,005. Checkpoint warm start
(epoch 0) tetap boleh terpilih. Fungsi ini hanya untuk pemilihan model, tidak dilaporkan sebagai
skor kompetisi. Enam profil IDSSE dan tiga profil satu game SkillCorner masuk validasi;
label test tidak masuk optimizer atau kriteria pemilihan epoch/campuran. Arsitektur dikembangkan dengan audit hasil test lama, sehingga evaluasinya berulang.

Epoch 12 dipilih: spellF1 validasi IDSSE 0,74421 (warm start 0,74389), akurasi parsial SkillCorner
96,36% (warm start 94,04%). Kenaikan skor IDSSE sangat kecil; manfaat utama yang terlihat pada
validasi adalah adaptasi provider. Laporan menyimpan seluruh riwayat loss, utilitas, kelayakan
checkpoint, game/sumber dan hash parameter.

## Pemeriksaan dan reproduksi

Empat tes loss memeriksa batas yang benar, pengabaian separator, batch seluruhnya X dan
mask dropout. Empat tes adapter memastikan tidak ada D buatan, acuan konflik dibuang,
anchor terakhir tetap terjaga, dan interpolasi tidak menyeberangi babak. Semua 238 tes lokal
lulus setelah penambahan ini; source/default/scorer panitia tetap dipertahankan.

```powershell
python tools/train_temporal_v2.py
python tools/select_temporal_ensemble.py
python tools/prepare_skillcorner.py
python tools/train_temporal_domain.py
python tools/select_domain_ensemble.py
python tools/evaluate_temporal.py --model-dir outputs/temporal_domain --mode test
python tools/evaluate_temporal.py --model-dir outputs/temporal_domain --mode anchors
python tools/evaluate_temporal.py --model-dir outputs/temporal_domain --mode competition
python tools/evaluate_skillcorner.py
```

Neural runtime memakai dependensi opsional `requirements-temporal.txt`; induk joint dan
data IDSSE disiapkan oleh `run_all_tech.cmd`. Data/parameter/laporan baru berada di direktori
lokal yang di-ignore. Tidak ada video/data kompetisi yang diunggah, label manusia BAR-ATH,
perubahan default submission, push atau pengiriman hasil.

## Hasil akhir dan batas target

| Model | Asli | Bola hilang | Hilang + jitter |
| --- | ---: | ---: | ---: |
| TCN awal | 0.7950 | 0.6905 | 0.6732 |
| TCN panjang + loss batas | 0.7838 | 0.6731 | 0.6549 |
| Adaptasi provider, TCN panjang | 0.7851 | 0.6756 | 0.6558 |
| Adaptasi provider, TCN kecil | 0.7975 | 0.6916 | 0.6721 |
| Refit akhir, lima game IDSSE | 0.7975 | 0.6867 | 0.6788 |

Semua hasil di tabel adalah rerata dua game IDSSE yang sudah terekspos sebelumnya.
Adaptasi TCN kecil memilih epoch 4 dengan tujuan memaksimalkan spellF1 validasi IDSSE,
dengan penurunan akurasi parsial provider dibatasi 0,005. Skor aslinya 0,79083/0,80419.
Refit akhir memakai checkpoint adaptasi kecil, seluruh lima game train+validasi lama
(715.116 frame berlabel kuat), dua game SkillCorner training dengan label parsial, dan
jadwal tetap 10 epoch. Dua game test IDSSE tetap dikecualikan. Normalisasi dan decoder tetap.
Tidak ada pemilihan checkpoint atau skor validasi baru pada lima game yang kini dipakai fitting.
Protokol jadwal disimpan sebelum melihat hasil refit di `outputs/temporal_v2_workflow/final_refit_policy.json`.

Skor refit asli 0,79458/0,80042. Nilai terbaik yang diamati sekitar 0,7975, masih <0,8.
Kenaikan terhadap TCN awal hanya sekitar 0,0025; ini tidak membuktikan keunggulan statistik.
Refit menurunkan hasil bola hilang saja sekaligus menaikkan hasil hilang+jitter. Tidak ada
kandidat yang dinyatakan unggul universal atau menjadi submission otomatis.
Pengembangan memakai audit hasil test lama; pemilihan epoch/campuran tidak memasukkan
label test ke optimizer atau menghitung kriterianya dari test. Ini tetap evaluasi berulang.

## Benchmark provider tambahan

| Model | H/A parsial asli | Bola hilang | Hilang + jitter |
| --- | ---: | ---: | ---: |
| TCN awal | 96.23% | 94.99% | 94.88% |
| TCN panjang + loss batas | 96.82% | 96.28% | 95.75% |
| Adaptasi provider, TCN panjang | 98.14% | 97.66% | 97.46% |
| Adaptasi provider, TCN kecil | 96.96% | 96.38% | 96.26% |
| Refit akhir, lima game IDSSE | 97.21% | 96.80% | 96.65% |

HMM teratur pada input asli mendapat 92.41%. Acuan mencakup 30,183 titik pemilik diketahui.
Ini kecocokan terhadap silver labels provider pada game 1927964, bukan akurasi D,
spellF1 lengkap, atau skor BAR-ATH. Tidak dihitung interval binomial atas frame yang
saling berkorelasi seolah-olah merupakan ribuan game independen.

Evaluasi awal membandingkan TCN awal, TCN panjang dan adaptasi panjang setelah semua model
tersebut dibekukan. Pengembangan adaptasi kecil/refit merupakan gelombang berikutnya;
game provider tersebut kemudian sudah terekspos. Frame/label game itu tetap tidak masuk
fitting atau pemilihan epoch, tetapi hasil lima model tidak diklaim berasal dari satu
holdout yang seluruhnya belum pernah dilihat. Laporan awal beserta snapshot evaluator
disimpan sebagai `outputs/skillcorner_benchmark/report_initial_models.json` dan
`evaluator_initial_models.py`. Laporan gabungan akhir berisi 16 evaluasi parsial.

Transfer Metrica tetap belum mengungguli HMM teratur 96,24%/97,93% pada dua sample game.
Dengan demikian hasil SkillCorner yang lebih tinggi belum membuktikan transfer universal.

Ensemble TCN awal + adaptasi panjang mendapat validasi 0,74777. Gain terhadap komponen
terbaik sekitar 0,00355, di bawah margin adopsi 0,005. Dua ensemble disimpan sebagai riset
yang tidak diprioritaskan dan tidak dievaluasi pada test.

## Artefak dan pemeriksaan akhir

Laporan `outputs/temporal_v2_workflow/final_verification.json` memverifikasi lima model,
30 CSV test lengkap, sepuluh hasil anchor, seluruh CSV penuh/receipt/model/sumber, mask
babak dan dua ensemble yang ditolak. Model refit tidak mengklaim validasi baru. Lima CSV
penuh sudah divalidasi format, diulang identik dan disalin ke `evaluation/results/`.
238 tes, lint/format 98 file dan sembilan pemeriksaan utama lulus. Pemeriksaan utama
tetap memeriksa artefak joint; verifikasi kandidat neural dilakukan terpisah seperti di atas.

| Kandidat CSV | Possession | <2 detik | D | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| exp_temporal_domain_v1.csv | 456 | 53 | 26.18% | c19813ba1403ee976f3f6ed3778ce48b05378f9b3c710c7a8990fce64d03a4bb |
| exp_temporal_refit.csv | 452 | 49 | 28.49% | e0087dc6d6715359a9283161ea5c2ac345ec2e671f4dacb5f6cd1044420e229f |

Statistik possession/D bukan akurasi BAR-ATH; acuan penuh pertandingan kompetisi belum tersedia.
Klik `run_domain.cmd` untuk mereproduksi keluarga eksperimen setelah alur TCN awal siap.
Prioritas berikutnya adalah menambah game lain dengan label interval H/A/D lengkap dan
menyisihkan game baru sebelum pengembangan berikutnya. Dataset tambahan sekarang hanya
memberikan acuan H/A parsial, sehingga tidak cukup untuk mengonfirmasi seluruh target 0,8.
