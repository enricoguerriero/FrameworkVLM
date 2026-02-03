import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


class ResusVideoLabeler:
    _PATTERNS: Dict[str, re.Pattern] = {
        "Baby visible": re.compile(r"baby\s*visible", re.I),
        "Ventilation" : re.compile(r"(?:^|[_\-])P?-?(?:CPAP|PPV)(?:[_\-]|\.|$)", re.I),
        "Stimulation" : re.compile(r"(?:^|[_\-])P?-?Stimulation\s*(?:backnates|trunk|extremities)", re.I),
        "Suction"     : re.compile(r"(?:^|[_\-])P?-?Suction(?:[_\-]|\.|$)", re.I),
    }
    ORDER = ["Baby visible", "Ventilation", "Stimulation", "Suction"]

    @classmethod
    def labels_for_file(cls, fname: str | Path) -> Set[str]:
        name = Path(fname).name
        return {lbl for lbl, pat in cls._PATTERNS.items() if pat.search(name)}

    @classmethod
    def encode_multi_hot(cls, fname: str | Path) -> List[int]:
        present = cls.labels_for_file(fname)
        return [int(lbl in present) for lbl in cls.ORDER]

def extract_case_number(filename: str | Path) -> str | None:
    # Adjust the regex if your filename pattern changes!
    m = re.search(r"Resuscitation_(\d{3})_", str(filename))
    return m.group(1) if m else None


def build_dataset(root: Path, case_numers: List[str], copy_to: Path | None = None) -> pd.DataFrame:
    video_paths = sorted(root.rglob("*.avi"))
    records = []

    for idx, vid in enumerate(video_paths):
        case_number = extract_case_number(vid)
        if case_number not in case_numers:
            continue

        # labels = ResusVideoLabeler.labels_for_file(vid)
        multi_hot = ResusVideoLabeler.encode_multi_hot(vid)
        labels_dict = dict(zip(
            ["baby_visible","ventilation","stimulation","suction"],
            multi_hot
        ))
        dest = None

        if copy_to is not None:
            copy_to.mkdir(parents=True, exist_ok=True)
            dest = copy_to / f"{idx:06d}{vid.suffix.lower()}"
            shutil.copy2(vid, dest)

        records.append({
            "video_path": str(dest if dest else vid),
            **labels_dict
        })

    return pd.DataFrame.from_records(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path,
                   help="Directory containing raw *.avi files (walked recursively)")
    p.add_argument("--out",  required=True, type=Path,
                   help="Path to write the output csv file")
    p.add_argument("--set", type=str, required=True,
                   help="Train, Validation or Test?")
    p.add_argument("--copy_videos_to", type=Path,
                   help="(Optional) folder to receive a clean, sequential copy of every video")
    args = p.parse_args()

    # HARDCODED CODES FOR DATASETS
    train_idxs = ["001", "002", "003", "004", "005", "006", "010", "011", "015", "016", "017"]
    validation_idxs = ["007", "008", "009"]
    test_idxs = ["012", "013", "014"]

    if args.set.lower() == "train":
        case_numbers = train_idxs
    elif args.set.lower() == "validation":
        case_numbers = validation_idxs
    elif args.set.lower() == "test":
        case_numbers = test_idxs
    else:
        raise ValueError("Unknown set: must be train, validation, or test")

    df = build_dataset(args.root, case_numbers, args.copy_videos_to)

    out_path: Path = args.out
    if out_path.suffix.lower() != ".csv":
        out_path = out_path.with_suffix(".csv")

    df.to_csv(out_path, index=False)
    print(f"✓ Wrote {len(df):,} records to {out_path}")


if __name__ == "__main__":
    main()