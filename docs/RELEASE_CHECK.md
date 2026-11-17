# Pemeriksaan kode dan artefak eksternal

Diperbarui 17 November 2026 setelah pengeluaran jalur bola video dan training HMM BAR-ATH.

```powershell
python tools/check_repository_content.py
python tools/check_release.py
```

Perintah pertama memeriksa daftar file Git terhadap `.github/excluded-paths.txt` dan larangan
aset privat. Perintah kedua memeriksa kode panitia/baseline, daftar pengecualian, lint, format,
tes sintetis, dan dependensi. Hasilnya ditulis ke ignored `outputs/release_check/`.
Tidak diperlukan dataset atau checkpoint untuk pemeriksaan kode.

Berkas yang dikeluarkan tetap disimpan pada arsip lokal di `outputs/`. CSV, bobot, cache,
atau draft submission lama tidak menjadi keluaran yang didukung oleh snapshot terbaru ini.
Repo baru memakai riwayat baru. Arsip dan commit penelitian yang dikeluarkan hanya ada
pada workspace/repository lama; tidak diimpor ke repo baru ini.

## Memeriksa artefak model eksternal

Untuk penelitian pohon/TCN yang dipertahankan, bangun ulang checkpoint dan laporan melalui
`run_all_tech.cmd`, `run_temporal.cmd`, `run_domain.cmd`, atau skrip riset/refit yang sesuai.
Training memakai pertandingan eksternal. Jangan melatih atau mengkalibrasi model pada BAR-ATH.
Perubahan sumber dan config mengubah hash provenance; laporan lama harus diregenerasi,
bukan diedit supaya lolos pemeriksaan.

```powershell
python tools/check_release.py --with-artifacts
```

Opsi ini menambahkan verifikasi benchmark anchor eksternal, eksperimen tim, dan kandidat
supervised pada benchmark H/A/D lengkap. Kandidat/plan/lokasi benchmark bisa diberikan melalui
`--supervised-config`, `--supervised-plan`, dan `--supervised-benchmark-dir`.
Pemeriksa tidak melatih atau memperbaiki artefak secara otomatis. Kelulusan pemeriksaan kode
maupun artefak tidak membuktikan akurasi BAR-ATH atau target spellF1 0,8.

## CI dan review

PR ke `dev` menjalankan lint, format, tes, larangan berkas, serta validator dan scorer baseline
pada Windows dan Ubuntu. Job agregat `checks` wajib lulus. Review anggota tim tetap diperlukan
sebelum merge. Kode panitia di `evaluation/`, `starter/`, dan `games.py` tetap dipertahankan.
