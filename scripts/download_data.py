"""Fetch the nea 5-fold split ids and check that the ASAP TSV is in place.

The ASAP essays themselves come from Kaggle (see README); this script cannot download them for you.
"""
import time
import urllib.request

from aes.data import FOLDS_DIR, load_asap, load_fold, tsv_path

NEA = "https://raw.githubusercontent.com/nusnlp/nea/master/data"


def fetch(url, dest, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                dest.write_bytes(r.read())
            return
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def main():
    for k in range(5):
        d = FOLDS_DIR / f"fold_{k}"
        d.mkdir(parents=True, exist_ok=True)
        for split in ("train", "dev", "test"):
            f = d / f"{split}_ids.txt"
            if not f.exists() or f.stat().st_size == 0:
                fetch(f"{NEA}/fold_{k}/{split}_ids.txt", f)
    print(f"folds ready in {FOLDS_DIR}")

    if not tsv_path().exists():
        print(f"Missing {tsv_path()}. Download it from Kaggle (README section 'Data'), then rerun.")
        return
    df = load_asap()
    print(f"{len(df)} essays, {df.prompt.nunique()} prompts")
    for k in range(5):
        f = load_fold(k)
        print(f"fold {k}: " + ", ".join(f"{s}={len(v)}" for s, v in f.items()))


if __name__ == "__main__":
    main()
