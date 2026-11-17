<!-- Base PR harus `dev`. Hanya PR rilis yang boleh ke `main` (dari `dev`). -->

## Apa yang diubah

<!-- Satu ide per PR. Jelaskan dalam 1–3 kalimat, dan sebut issue-nya: "Closes #12" -->

## Hasil sebelum vs sesudah

<!-- "Sebelum" = dev saat ini. Publik hanya cek correctness, bukan dasar memilih model.
Label manusia BAR-ATH dilarang untuk training/validasi. Jangan menilai terhadap pseudo-label sendiri.
Jika tidak ada acuan independen yang diizinkan, tulis "belum diukur". -->

| | Acuan dan cakupan evaluasi yang diizinkan | Metrik | Possession penuh | Durasi rata-rata (s) |
|---|---|---|---|---|
| Sebelum (dev) | | | | |
| Sesudah (PR ini) | | | | |

Perintah untuk mereproduksi:

```
python -m mtah.predict --name ... --set ...
```

## Checklist

- [ ] `pytest` hijau di laptop saya
- [ ] Angka yang bisa di-tuning ada di `config/`, bukan di kode
- [ ] Tidak ada data pertandingan, cache, atau output yang ikut ter-commit
- [ ] Pseudo-label ditandai sebagai keluaran algoritma, bukan ground truth independen
- [ ] Kalau menambah library: sudah dipin di `requirements-team.txt`
- [ ] Kalau mengubah aturan possession: `docs/ATURAN_POSSESSION.md` ikut diperbarui

## Untuk reviewer

<!-- Bagian mana yang paling perlu dicek? Cantumkan perintah reproduksi dan batas evaluasi.
Untuk kandidat HMM terlatih: python tools/check_release.py (lihat docs/RELEASE_CHECK.md). -->
