# Cara kerja tim

Ada tiga jenis branch:

```
budi/bola-keluar-garis ─┐
sari/possession-kontrol ─┼─ PR (squash) ──> dev ── PR rilis (merge commit) ──> main
nizar/laporan-durasi ───┘                  staging                         versi stabil / submission
```

| Branch | Isinya | Siapa yang mengubah |
| --- | --- | --- |
| `<nama>/<topik>` | Satu ide yang sedang dikerjakan | Pembuatnya, bebas push |
| `dev` | Staging: gabungan semua ide yang sudah di-review. Branch default repo | Hanya lewat PR + 1 approve + CI hijau |
| `main` | Versi stabil. File submission final dibuat dari sini | Hanya PR dari `dev`, + 1 approve + CI hijau |

Push langsung ke `dev` atau `main` ditolak oleh GitHub (ruleset). PR ke `main` dari branch selain
`dev` ditolak oleh CI.

## Alur harian (anggota tech)

```powershell
# 1. Mulai dari dev terbaru
git switch dev
git pull

# 2. Branch baru untuk satu ide. Nama: <nama>/<topik>
git switch -c budi/bola-keluar-garis

# 3. Kerjakan, lalu jalankan dan ukur
python -m mtah.predict --name exp_bola_keluar --set dead_ball.method=out_of_bounds
pytest

# 4. Simpan dan kirim
git add -A
git commit -m "dead_ball: bola di luar garis dianggap mati"
git push -u origin budi/bola-keluar-garis

# 5. Buka PR ke dev (atau lewat tombol hijau di halaman GitHub)
gh pr create --base dev --web
```

`--web` membuka halaman PR di browser dengan deskripsi yang sudah terisi template
(`.github/pull_request_template.md`), jadi tinggal mengisi bagian yang kosong. Membuat PR lewat
tombol *Compare & pull request* di GitHub juga otomatis memakai template yang sama. Pastikan base PR adalah **`dev`**.

Isi tabel "Hasil sebelum vs sesudah" di template PR. Angkanya dari output `mtah.predict`, untuk
`dev` dan untuk branch kamu. Tempel juga baris log eksperimen ke Google Sheet log.

## Review

Setiap PR butuh **1 approve dari anggota tech lain**. Pembuat PR tidak bisa meng-approve PR-nya
sendiri, dan push baru setelah approve akan membatalkan approve tersebut.

Yang dicek reviewer:

1. Tabel hasil terisi, dan angka di pertandingan penuh masuk akal. Contohnya: jumlah possession
   tidak tiba-tiba berlipat, dan durasi rata-rata tidak anjlok.
2. Angka yang bisa di-tuning ada di `config/`, bukan di dalam kode.
3. Method lama tetap ada. Ide baru ditambahkan sebagai method baru, bukan menimpa yang lama.
4. Tidak ada perubahan di `evaluation/`, `starter/`, atau `games.py` (milik panitia).
5. Kodenya bisa dipahami. Kalau reviewer tidak paham, minta penjelasan di komentar.

Setelah approve dan CI hijau, **pembuat PR** yang menekan **Squash and merge**: satu ide jadi satu
commit di `dev`. Branch otomatis terhapus.

## Rilis: dev ke main (Tech 1)

Dilakukan di akhir setiap minggu (target mingguan di dokumen rencana) dan sekali lagi untuk
submission final, setelah `dev` dicek dengan `python tools/cek_setup.py` dan `mtah.predict`.
Jalankan `python tools/check_repository_content.py` dan `python tools/check_release.py`.
Gunakan `--with-artifacts` setelah membangun ulang model dan laporan eksternal yang relevan;
lihat `docs/RELEASE_CHECK.md`. Daftar metode yang dipertahankan tersedia di README.
Training dan kalibrasi model hanya memakai pertandingan eksternal. File yang dikeluarkan
dalam `.github/excluded-paths.txt` tidak boleh ditambahkan kembali.

```powershell
gh pr create --base main --head dev --title "Rilis minggu 1" --web
```

Review seperti biasa, lalu gabungkan dengan **Create a merge commit** (bukan squash), supaya
riwayat `dev` dan `main` tetap sejalan.

Repository dikelola dalam organisasi `Goalorithm`. Commit pekerjaan ini memakai akun
`Affandi21`. Persetujuan PR tetap diberikan anggota tim lain; pembuat PR tidak
meng-approve pekerjaannya sendiri.

## Aturan yang tidak bisa ditawar

- **Data pertandingan tidak pernah di-commit**, termasuk turunannya (`cache/`, `outputs/`, video,
  `.npz`). Repo ini akan dijadikan publik saat submit, dan riwayat git menyimpan semuanya.
- **Label submission keluar dari algoritma.** Aturan upstream terbaru melarang label manusia
  BAR-ATH untuk training maupun validasi. Alur lama di `labels/` dinonaktifkan; lihat
  `docs/RULE_UPDATE.md`. Pseudo-label algoritma disimpan di `outputs/`, bukan ground truth.
- **Jangan memilih versi berdasarkan jendela publik 5 menit atau pseudo-label sendiri.**
  Gunakan pemilihan tanpa label yang didokumentasikan, acuan game lain yang diizinkan,
  atau evaluasi panitia. Benchmark anchor umpan belum cukup memilih pemenang spellF1.
- **Hasil harus bisa direproduksi.** Tidak ada unsur acak, atau seed dikunci di config.

## Bekerja dengan config

```powershell
# satu nilai, tanpa file baru
python -m mtah.predict --name exp_w75 --set possession.window=75

# banyak nilai: salin config, ubah, jalankan
copy config\default.toml config\exp_x.toml
python -m mtah.predict --config config\exp_x.toml --name exp_x
```

Config yang menang dipindahkan ke `config/default.toml` lewat PR. Itulah config yang
menghasilkan file final.

## Menambah method baru

Contoh di `src/mtah/dead_ball.py`:

```python
def out_of_bounds(tracking, features, tagging, cfg):
    margin = float(cfg.get("margin_m", 0.5))
    ...
    return dead          # bool array, panjang n_frames

METHODS = {
    "no_ball_window": no_ball_window,
    "out_of_bounds": out_of_bounds,   # tambahkan di sini
}
```

Lalu pilih di config: `[dead_ball] method = "out_of_bounds"` dan `margin_m = 0.5`. Kontrak input
dan output tertulis di bagian atas `dead_ball.py` dan `possession.py`. Fitur bersama
(`features.py`) boleh ditambah, tetapi arti key yang sudah ada tidak boleh diubah.

## Alat bantu

| Perintah | Untuk apa |
| --- | --- |
| `python tools/cek_setup.py` | Cek laptop siap: data, cache, kecocokan dengan baseline |
| `python -m mtah.predict` | Jalankan pipeline, validasi, skor, baris log |
| `python tools/timeline.py --window publik --pred A B` | Gambar GT vs prediksi vs sinyal bola |
| `python tools/frame_view.py --from F1 --to F2 --step 25` | Posisi pemain dan bola di frame tertentu |
| `python tools/match_report.py A B` | Statistik pertandingan penuh tanpa GT |
| `python tools/ai_labels.py` | Ekspor pseudo-label lokal, bukan ground truth |
| `python tools/check_release.py` | Periksa artefak terlatih, reproduksi dan kesiapan lokal |
| `python evaluation/run_evaluation.py` | Papan skor resmi panitia untuk semua file di `evaluation/results/` |

## Anggota bisnis (tanpa akun GitHub)

Anggota bisnis bekerja sepenuhnya di folder Drive tim dan grup WhatsApp:

| Kebutuhan | Tempat |
| --- | --- |
| Tugas dan statusnya | Spreadsheet "Goalorithm – Sheet Tim", tab `Tugas tim bisnis` |
| Rencana diagnostik | Tab `windows`; nama dev/holdout lama bukan ground truth |
| Aturan label terbaru | `docs/RULE_UPDATE.md`; panduan anotasi lama diarsipkan |
| Buku aturan possession | Google Doc "Buku Aturan Possession – Goalorithm" (pemilik Bisnis 1) |
| Draft dokumen teknis | Google Doc "Technical Document (draft) – Goalorithm" (pemilik Bisnis 2) |

Tech 1 menjembatani keduanya:

- Issue bisnis (#1, #3, #4, #6, #13, #14) di-assign ke Tech 1 dengan nama anggota bisnis di
  judul. Tech 1 memindahkan kartunya di board sesuai tab `Tugas tim bisnis`.
- Rencana jendela di `labels/windows.csv` boleh dipelihara. Jangan mengisi `perubahan.csv`
  dengan label manusia BAR-ATH atau menyalin pseudo-label sebagai acuan independen.
- Buku aturan disalin ke `docs/ATURAN_POSSESSION.md` dan dokumen teknis ke
  dokumentasi metode yang dipertahankan lewat PR, minimal setiap rilis mingguan.

## Menarik update dari panitia (Tech 1)

```powershell
git switch dev
git pull
git switch -c sync/upstream
# sekali saja per laptop:
# git remote add upstream https://github.com/metrica-sports/more-than-a-hack-2027.git
git fetch upstream
git merge upstream/main
# README.md bisa konflik: pertahankan versi tim, lalu cek perubahan panitia secara manual
git push -u origin sync/upstream
gh pr create --base dev --web
```
