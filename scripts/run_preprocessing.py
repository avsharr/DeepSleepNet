"""
Run data preprocessing: raw EDF -> npz (epochs with labels).
"""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preprocessing import process, RAW_DATA, PROCESSED_DATA


def main():
    files = os.listdir(RAW_DATA)

    # all psg files
    for psg in [f for f in files if f.endswith("PSG.edf")]:
        # looking for a pair hypnogram
        hyp_candidates = [f for f in files if f.startswith(psg[:7]) and "Hypnogram" in f]

        if hyp_candidates:
            process(
                os.path.join(RAW_DATA, psg),
                os.path.join(RAW_DATA, hyp_candidates[0]),
                os.path.join(PROCESSED_DATA, psg.replace(".edf", ".npz"))
            )


if __name__ == "__main__":
    main()
