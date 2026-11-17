# Pemeriksaan dataset dan video terbaru

**Pembaruan aturan:** alur anotasi manusia pada bagian riwayat di bawah sudah diarsipkan.
Aturan terbaru melarangnya termasuk untuk validasi; gunakan [RULE_UPDATE.md](RULE_UPDATE.md)
dan `tools/ai_labels.py`. Keenam jendela sudah diisi pseudo-label algoritma lokal, bukan manusia.

Pemeriksaan lokal pada 11 November 2026, sesudah penambahan dataset dan kedua video.

## Berkas dan penataan folder

Delapan folder unduhan memiliki awalan
`More Than A Hack 2027 - Initial Challenge - Dataset-20261111T034917Z-1-`.

| Akhiran folder | Isi |
| --- | --- |
| `003` | README dataset |
| `004` | Smart Tagging tactical |
| `005` | Metadata ATD broadcast |
| `006` | Metadata ATD tactical |
| `007` | Homografi tactical |
| `008` | Homografi broadcast |
| `009` | Tracking mentah broadcast |
| `010` | Tracking mentah tactical |

Video di folder utama adalah `BAR-ATH_tactical-002.mp4` (5.222.307.334 byte) dan
`BAR-ATH_broadcast-001.mp4` (13.962.360.320 byte). Empat berkas tactical yang baru diunduh
identik dengan unduhan sebelumnya berdasarkan SHA-256.

`tools/prepare_data.py` memeriksa hash sumber sebelum menautkan sembilan aset ke nama standar
di `data/tactical/` dan `data/broadcast/`, termasuk `BAR-ATH_tactical.mp4` dan
`BAR-ATH_broadcast.mp4`. Video ditautkan dengan hard link: tidak disalin, dipindahkan, diganti
nama, atau ditranscode. Folder tactical tetap memakai junction lokal yang sebelumnya dibuat;
folder sumber lama harus tetap tersedia. Semua tautan dan laporan tetap di-ignore oleh Git.

## Hasil pemeriksaan penuh

Kedua video didekode sampai akhir. Pemeriksaan tidak menyimpan atau mengunggah frame citra.

| Pemeriksaan | Tactical | Broadcast |
| --- | ---: | ---: |
| Video berhasil didekode | 199.319 frame | 164.487 frame |
| Tracking berurutan dari frame ID 1 | 199.319 rekaman | 164.488 rekaman |
| Slot tracking | 46 | 51 |
| Baris homografi berurutan dari indeks 0 | 199.319 | 164.488 |
| Homografi valid | 199.319 (100%) | 108.248 (65,81%) |
| FPS | 25 | 25 |
| Resolusi | 1920 x 1080 | 1920 x 1080 |
| Frame ditandai corrupt oleh decoder | 0 | 0 |
| Timestamp menyimpang dari urutan indeks / 25 | 0 | 0 |
| Kesesuaian jumlah frame video dengan referensi | Cocok | Selisih 1 frame |

Smart Tagging tactical memuat 638 interval. Tag kick-off kedua babak mencakup anchor panitia.
Seluruh matriks homografi yang bertanda valid memiliki nilai finite.

**Tactical siap sebagai referensi submission dan peninjauan label.** Selisih satu frame pada
broadcast sudah terkonfirmasi melalui dekode penuh, bukan sekadar header. Data tidak dipotong
atau ditambal untuk menyamarkan selisih ini. Broadcast belum dipakai oleh HMM dan tidak boleh
dianggap tersinkron dengan tactical.

Laporan lengkap: `outputs/data_audit/assets.json` dan `outputs/data_audit/validation.json`.
Audit semua sumber mengembalikan kode `2` karena mismatch broadcast; tactical sendiri lulus.

## Hasil pipeline dan pengujian

- Baseline tetap `Ready`: `0 frames differ`.
- HMM dijalankan ulang sebagai `outputs/exp_hmm_video_ready.csv`, dengan hasil identik.
- SHA-256: `48f403095696ac83a60a06787c1320eff2db5c8ca95a7f81ffc8c44752773e86`.
- Seluruh 108 posisi usulan berhasil dicapai dengan timestamp frame yang persis.
- 182 tes lulus; lint, format, dan konsistensi dependency diperiksa.

Seek memakai presentation timestamp, bukan perkiraan frame dari seek waktu yang dibulatkan.
Rujukan API: [PyAV Time](https://pyav.org/docs/stable/api/time.html).

## Peninjau lokal — dihapus

Aplikasi peninjau video dan pengimpor label manual dibuat sebelum pembaruan aturan `f8f2629`,
yang melarang label manusia pada BAR-ATH untuk tujuan apa pun. Keduanya sudah dihapus dari
repositori. Diagnostik yang tersisa adalah pseudo-label algoritmik di `outputs/ai_labels/`
(`tools/ai_labels.py`), yang bukan ground truth.

## Mengulang pemeriksaan

```powershell
python tools/prepare_data.py
python tools/audit_data.py --full-decode
```

Dekode penuh memerlukan beberapa menit. Semua hasil tetap lokal. Jangan commit data, video,
cache, prediksi, atau hasil turunan pertandingan.
