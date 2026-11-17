# Buku aturan possession tim Goalorithm

> **Pembaruan 11 November 2026:** jangan mengisi tabel dengan anotasi atau koreksi manual pada
> frame pertandingan ini. Aturan resmi terbaru melarang label manusia untuk training/validasi.
> Aturan umum possession tetap berguna; alur pencatatan contoh frame di bawah sudah diarsipkan.
> Lihat [RULE_UPDATE.md](RULE_UPDATE.md).

Pegangan bersama untuk tiga hal: cara melabeli validasi, cara kode memutuskan, dan bagian
"possession rules" di dokumen teknis. Pemilik: Bisnis 1. Setiap aturan diambil dari bukti di
ground truth publik (nomor frame), bukan dari intuisi saja.

**Yang sudah pasti dari panitia** (CHALLENGE.md):

- Hanya tiga keadaan saat dinilai: `H`, `A`, `D`. Tidak ada "direbutkan" atau "transisi".
- Satu possession = deretan terpanjang satu tim dengan bola hidup. `H H D D H H` adalah dua
  possession.
- Possession berpindah hanya kalau lawan **mengontrol** bola. Sentuhan tanpa kontrol, defleksi,
  dan duel udara tanpa kontrol sesudahnya tidak dihitung.
- Bola liar atau direbutkan: possession tetap milik tim terakhir.
- Ground truth diturunkan secara mekanis dari tracking yang sudah dikoreksi: `D` di frame yang
  ditandai bola mati, selain itu tim yang tercatat menguasai bola.

## Kasus-kasus

Isi kolom "Aturan kita" dan "Bukti" dari hasil kalibrasi. Status: `usulan`, `disepakati`, atau
`sudah di kode`.

| Kasus | Aturan kita | Bukti di GT publik (frame) | Status |
| --- | --- | --- | --- |
| Bola keluar garis samping: kapan D mulai | | | usulan |
| Lemparan ke dalam: kapan D berakhir dan siapa yang menguasai | | | usulan |
| Tendangan gawang | | | usulan |
| Sepak pojok | | | usulan |
| Peluit pelanggaran: kapan D mulai | | | usulan |
| Tendangan bebas: kapan D berakhir | | | usulan |
| Gol sampai kick-off | | | usulan |
| Sentuhan sekali lawan tanpa kontrol | | | usulan |
| Defleksi umpan | | | usulan |
| Duel udara | | | usulan |
| Bola liar lama di tengah | | | usulan |
| Kiper menangkap bola | | | usulan |
| Operan salah langsung ke lawan | | | usulan |

## Catatan kalibrasi

| Tanggal | Pelabel | Frame | Label kita | Ground truth | Pelajaran |
| --- | --- | --- | --- | --- | --- |
| | | | | | |
