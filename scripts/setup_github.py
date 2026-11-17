"""One-time GitHub setup for the team repository, using the GitHub CLI (gh).

Run it ONCE, logged in to gh as the repository OWNER (the account that owns the repo, because
only an admin can change settings and rulesets):

    gh auth login                              # choose the owner account
    gh auth refresh -s project                 # only if you also want the project board
    python scripts/setup_github.py --repo OWNER/REPO --invite USER1 USER2 ... --project

What it does (each step is skipped or updated if already done; --dry-run changes nothing):
  1. repository settings: squash merge and merge commits allowed, branches deleted after merge
  2. creates the `dev` staging branch from main and makes it the default branch
  3. rulesets from .github/ruleset-dev.json and ruleset-main.json: no direct pushes, every change
     through a PR with 1 approval from someone else, review threads resolved, CI ("checks")
     green; feature -> dev by squash, dev -> main by merge commit
  4. labels used by the board and the issues
  5. invites collaborators with write access
  6. with --project: a GitHub Projects board, linked to the repo
  7. the first issues (setup week + first tasks of each module), on the board if there is one
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    "tech": ("1d76db", "Pekerjaan kode"),
    "bisnis": ("fbca04", "Pekerjaan tim bisnis"),
    "bola-mati": ("5319e7", "Modul dead_ball.py"),
    "possession": ("0e8a16", "Modul possession.py"),
    "infra": ("c5def5", "Repo, alat bantu, evaluasi"),
    "label-validasi": ("f9d0c4", "Diagnostik pseudo-label; bukan ground truth BAR-ATH"),
    "dokumen": ("bfdadc", "Dokumen teknis dan buku aturan"),
    "eksperimen": ("d4c5f9", "Ide untuk menaikkan skor"),
    "rilis": ("b60205", "Submission akhir"),
    "admin": ("ededed", "Pendaftaran, form, administrasi"),
}

ISSUES = [
    (
        "Pastikan 5 anggota terdaftar dengan nama tim yang sama persis",
        "Batas: sebelum 10 Nov. Cek ke setiap anggota; kalau ragu, tanya panitia di "
        "hackathon.talentarena@mobileworldcapital.com.",
        ["bisnis", "admin"],
    ),
    (
        "Setup laptop sampai `python tools/cek_setup.py` semua OK",
        "Ikuti docs/SETUP.md. Satu komentar per anggota tech di issue ini setelah OK.",
        ["tech"],
    ),
    (
        "Audit dataset dan video tactical secara lokal",
        "Ikuti docs/DATA_AUDIT.md. Jangan unggah data/video kompetisi ke layanan AI cloud.",
        ["bisnis", "label-validasi"],
    ),
    (
        "Periksa perubahan aturan dan provenance pseudo-label",
        "Baca docs/RULE_UPDATE.md. Label manusia BAR-ATH dilarang, termasuk untuk validasi. "
        "Pseudo-label harus berasal dari algoritma lokal dan tidak dipakai sebagai ground truth.",
        ["bisnis", "label-validasi"],
    ),
    (
        "Periksa 6 jendela diagnostik di labels/windows.csv",
        "3 di babak 1 (frame 11185-90651), 3 di babak 2 (110808-190197), di luar jendela publik. "
        "4 dev, 2 holdout adalah nama area lama, bukan validation ground truth. "
        "Masing-masing 7500 frame (5 menit). Jangan lakukan anotasi manusia BAR-ATH.",
        ["tech", "infra"],
    ),
    (
        "Dokumentasikan definisi possession dan batas evaluasi",
        "Gunakan definisi resmi CHALLENGE.md dan docs/RULE_UPDATE.md. Jangan membuat label "
        "kalibrasi manusia atau memilih model berdasarkan skor jendela publik.",
        ["bisnis", "dokumen"],
    ),
    (
        "Catat 5 kesalahan bola mati dari baseline di jendela publik",
        "`python tools/timeline.py --window publik --pred baseline_smooth50`, lalu "
        "tools/frame_view.py di sekitar stoppage yang terlewat. Tulis temuannya di issue ini.",
        ["tech", "bola-mati"],
    ),
    (
        "Catat 5 kesalahan pergantian possession dari baseline",
        "Sama seperti issue bola mati, fokus pada possession yang terpecah atau terlambat.",
        ["tech", "possession"],
    ),
    (
        "dead_ball v1: bola keluar garis lapangan",
        "Method baru di src/mtah/dead_ball.py: posisi bola terakhir di luar garis -> D sampai "
        "bola terlihat lagi di dalam lapangan.",
        ["tech", "bola-mati", "eksperimen"],
    ),
    (
        "dead_ball: pakai set piece dari Smart Tagging",
        "Set piece = restart setelah stoppage. Ingat: tag Team untuk throw-in selalu N/A.",
        ["tech", "bola-mati", "eksperimen"],
    ),
    (
        "possession v1: possession pindah hanya saat lawan benar-benar mengontrol bola",
        "State machine / hysteresis: butuh beberapa frame penguasaan berturut-turut sebelum "
        "berpindah tim. Lihat CHALLENGE.md, 'Possession rules'.",
        ["tech", "possession", "eksperimen"],
    ),
    (
        "possession: kapan menulis X",
        "X di tengah possession gratis untuk spellF1, mahal di titik pergantian. Lihat "
        "EVALUATION.md, 'How an abstention is scored'.",
        ["tech", "possession", "eksperimen"],
    ),
    (
        "Draft dokumen teknis",
        "Tulis dokumen teknis kandidat yang sesuai aturan (bahasa Inggris), 5 bagian wajib.",
        ["bisnis", "dokumen"],
    ),
    (
        "Tes reproduksi dari README di laptop yang belum pernah dipakai",
        "Hanya berbekal README.md. Hasil sha256 harus sama dengan file final.",
        ["bisnis", "rilis"],
    ),
    (
        "Final: freeze, repo publik, submit form",
        "Checklist 'Checklist aturan dan submission' di dokumen rencana tim.",
        ["tech", "rilis"],
    ),
]


def gh(args, dry_run, check=True, capture=False):
    print("  $ gh " + " ".join(args))
    if dry_run:
        return ""
    result = subprocess.run(["gh", *args], text=True, capture_output=True)
    if check and result.returncode != 0:
        print(result.stderr.strip())
        raise SystemExit(f"gh failed: {' '.join(args[:3])}")
    return result.stdout if capture else result


def apply_ruleset(repo, path, target, existing, dry):
    """Create the ruleset, or replace an existing one for the same branch.

    allowed_merge_methods (squash into dev, merge commit into main) is a newer API field; if
    GitHub rejects it, the ruleset is applied without it and the convention is kept by hand.
    """
    body = json.loads(path.read_text(encoding="utf-8"))
    same = [
        r
        for r in existing
        if r["name"] == body["name"] or r["name"].split(":")[0].strip() == target
    ]
    url = f"repos/{repo}/rulesets/{same[0]['id']}" if same else f"repos/{repo}/rulesets"
    method = "PUT" if same else "POST"
    for attempt in (body, _without_merge_methods(body)):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(attempt, handle)
        result = gh(["api", "-X", method, url, "--input", handle.name], dry, check=False)
        if dry or result.returncode == 0:
            return
        print(f"  retrying without allowed_merge_methods ({result.stderr.strip()[:120]})")
    raise SystemExit(f"could not apply {path.name}")


def _without_merge_methods(body):
    clone = json.loads(json.dumps(body))
    for rule in clone["rules"]:
        rule.get("parameters", {}).pop("allowed_merge_methods", None)
    return clone


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="OWNER/REPO")
    parser.add_argument("--invite", nargs="*", default=[], help="GitHub usernames to invite")
    parser.add_argument("--project", action="store_true", help="also create the project board")
    parser.add_argument("--no-issues", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    owner = args.repo.split("/")[0]
    dry = args.dry_run

    print("1. repository settings")
    gh(
        [
            "api",
            "-X",
            "PATCH",
            f"repos/{args.repo}",
            "-F",
            "allow_squash_merge=true",
            "-F",
            "allow_merge_commit=true",
            "-F",
            "allow_rebase_merge=false",
            "-F",
            "delete_branch_on_merge=true",
        ],
        dry,
    )

    print("2. dev branch (staging), made the default branch")
    has_dev = gh(["api", f"repos/{args.repo}/branches/dev"], dry, check=False)
    if dry or has_dev.returncode != 0:
        sha = gh(
            ["api", f"repos/{args.repo}/git/ref/heads/main", "--jq", ".object.sha"],
            dry,
            capture=True,
        ).strip()
        gh(
            [
                "api",
                "-X",
                "POST",
                f"repos/{args.repo}/git/refs",
                "-f",
                "ref=refs/heads/dev",
                "-f",
                f"sha={sha or '<main sha>'}",
            ],
            dry,
        )
    else:
        print("  dev already exists")
    gh(["api", "-X", "PATCH", f"repos/{args.repo}", "-f", "default_branch=dev"], dry)

    print("3. rulesets on dev and main")
    existing = json.loads(gh(["api", f"repos/{args.repo}/rulesets"], dry, capture=True) or "[]")
    for filename, target in (("ruleset-dev.json", "dev"), ("ruleset-main.json", "main")):
        apply_ruleset(args.repo, ROOT / ".github" / filename, target, existing, dry)

    print("4. labels")
    for name, (colour, description) in LABELS.items():
        gh(
            [
                "label",
                "create",
                name,
                "--repo",
                args.repo,
                "--color",
                colour,
                "--description",
                description,
                "--force",
            ],
            dry,
        )

    print("5. collaborators")
    for user in args.invite:
        gh(
            [
                "api",
                "-X",
                "PUT",
                f"repos/{args.repo}/collaborators/{user}",
                "-f",
                "permission=push",
            ],
            dry,
        )

    project_title = None
    if args.project:
        print("6. project board")
        project_title = f"{args.repo.split('/')[1]} board"
        listing = gh(["project", "list", "--owner", owner, "--format", "json"], dry, capture=True)
        projects = json.loads(listing or '{"projects": []}')["projects"]
        found = [p for p in projects if p["title"] == project_title]
        if found:
            number = found[0]["number"]
        else:
            created = gh(
                [
                    "project",
                    "create",
                    "--owner",
                    owner,
                    "--title",
                    project_title,
                    "--format",
                    "json",
                ],
                dry,
                capture=True,
            )
            number = json.loads(created or '{"number": 0}')["number"]
        gh(
            ["project", "link", str(number), "--owner", owner, "--repo", args.repo],
            dry,
            check=False,
        )

    if not args.no_issues:
        print("7. first issues")
        listing = gh(
            [
                "issue",
                "list",
                "--repo",
                args.repo,
                "--state",
                "all",
                "--limit",
                "200",
                "--json",
                "title",
            ],
            dry,
            capture=True,
        )
        have = {i["title"] for i in json.loads(listing or "[]")}
        for title, body, labels in ISSUES:
            if title in have:
                continue
            with tempfile.NamedTemporaryFile(
                "w", suffix=".md", delete=False, encoding="utf-8"
            ) as handle:
                handle.write(body)
            command = [
                "issue",
                "create",
                "--repo",
                args.repo,
                "--title",
                title,
                "--body-file",
                handle.name,
            ]
            for label in labels:
                command += ["--label", label]
            if project_title:
                command += ["--project", project_title]
            gh(command, dry)

    print("\nDone. Check Settings > Rules > Rulesets on GitHub: the ruleset must say 'Active'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
