# Setup laptop dari nol (anggota tech)

Target akhir: `python tools/cek_setup.py` menampilkan **OK** di setiap baris. Perkiraan waktu
sekitar 30–45 menit, sebagian besar untuk mengunduh. Perintah di bawah untuk **Windows
PowerShell**; versi macOS/Linux ada di bagian paling bawah.

## 1. Pasang aplikasi

```powershell
winget install Python.Python.3.11
winget install Git.Git
winget install Microsoft.VisualStudioCode
winget install GitHub.cli
```

Tutup lalu buka lagi PowerShell, kemudian cek:

```powershell
py -3.11 --version     # Python 3.11.x
git --version
gh --version
```

Harus **3.11**, bukan 3.12 atau 3.13. Versi library dikunci untuk 3.11 supaya file submission
identik di semua laptop.

## 2. Akses ke repo

1. Terima undangan repo dari email atau dari https://github.com/notifications.
2. Login GitHub CLI dengan akun yang diundang: `gh auth login` (pilih GitHub.com, HTTPS, login
   lewat browser).
3. Atur identitas git (sekali saja):

```powershell
git config --global user.name "Nama Kamu"
git config --global user.email "email-yang-terdaftar-di-github@contoh.com"
```

## 3. Clone dan pasang environment

```powershell
cd C:\Kuliah            # folder mana saja, asal path-nya tanpa karakter aneh
git clone https://github.com/Goalorithm/mtah2027-Goalorithm.git
cd mtah2027-Goalorithm

py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

Kalau `Activate.ps1` ditolak dengan pesan *running scripts is disabled*, jalankan sekali
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, lalu ulangi.

Setiap membuka terminal baru untuk repo ini, aktifkan lagi dengan `.venv\Scripts\Activate.ps1`.
Tanda aktif: ada `(.venv)` di awal baris prompt.

Kalau muncul `ModuleNotFoundError: No module named 'mtah'` (atau `numpy`), berarti `.venv` belum
aktif atau `pip install -e .` belum dijalankan di dalamnya. Cek dengan
`python -c "import sys; print(sys.executable)"`: hasilnya harus berakhir dengan
`.venv\Scripts\python.exe`. Kalau tidak, aktifkan `.venv` lalu ulangi `pip install -e .`.
Di VS Code, pilih juga interpreter `.venv` (langkah 6), supaya terminal barunya otomatis aktif.

## 4. Data pertandingan

Unduh dari link Google Drive panitia, folder `tactical`, minimal tiga file ini (sekitar 245 MB):

- `BAR-ATH_tactical_atd.xml`
- `BAR-ATH_tactical_atd.txt`
- `BAR-ATH_tactical_smart_tagging.xml`

Letakkan di `data\tactical\` di dalam repo, sehingga ada file
`data\tactical\BAR-ATH_tactical_atd.txt`. Hati-hati dengan folder bertumpuk dari ZIP
(`data\tactical\tactical\...`). Video (4,86 GB) dan homografi tidak wajib untuk menjalankan
pipeline.

Folder `data\` sudah di-ignore oleh git, dan pre-commit memblokir commit file data. Data tidak
boleh keluar dari laptop kalian ke tempat lain.

## 5. Cek semuanya

```powershell
python tools/cek_setup.py
```

Run pertama mem-parsing tracking (sekitar 1–2 menit) lalu menyimpannya ke `cache\`. Run
berikutnya hanya beberapa detik. Baris terpenting:

```
OK    default config reproduces baseline_smooth50 (0 frames differ)
```

Artinya pipeline tim menghasilkan label yang sama persis dengan baseline panitia, jadi loader,
alignment frame, dan fitur sudah benar. Kalau ada baris FAIL, ikuti petunjuk di bawahnya, atau
tempel output-nya di issue "Setup laptop".

## 6. VS Code dan Claude Code

1. Buka repo: `code .`
2. Klik **Install** saat VS Code menawarkan ekstensi yang direkomendasikan: Python, Jupyter,
   Ruff, dan Claude Code.
3. `Ctrl+Shift+P` → *Python: Select Interpreter* → pilih `.venv`.
4. Claude Code: buka panelnya dan login. Claude Code otomatis membaca `CLAUDE.md` di root repo,
   yang berisi aturan proyek ini (jangan commit data, angka di `config/`, satu ide per PR, dan
   seterusnya).

Kode hasil bantuan AI tetap harus dipahami oleh yang membuat PR, karena reviewer akan bertanya.

## 7. Coba alurnya sekali

```powershell
python -m mtah.predict --name coba
python tools/timeline.py --window publik --pred baseline_smooth50 coba
```

Gambar timeline ada di `outputs\timeline_11185_18684.png`. Setelah itu baca
[CONTRIBUTING.md](../CONTRIBUTING.md) untuk alur branch dan PR. Clone otomatis berada di branch
`dev`, branch tempat semua pekerjaan dimulai.

---

### macOS / Linux

```bash
# Python 3.11 lewat pyenv, brew (python@3.11), atau package manager distro
git clone https://github.com/Goalorithm/mtah2027-Goalorithm.git
cd mtah2027-Goalorithm
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e . && pre-commit install
python tools/cek_setup.py
```
