import os
import requests
from tqdm import tqdm

# resolve paths: project root (parent of scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "data", "raw")

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"

# List of files from the article
FILENAMES = [
    "SC4001E0-PSG.edf",
    "SC4001EC-Hypnogram.edf",
    "SC4002E0-PSG.edf",
    "SC4002EC-Hypnogram.edf",
    "SC4011E0-PSG.edf",
    "SC4011EH-Hypnogram.edf",
    "SC4012E0-PSG.edf",
    "SC4012EC-Hypnogram.edf",
    "SC4021E0-PSG.edf",
    "SC4021EH-Hypnogram.edf",
    "SC4022E0-PSG.edf",
    "SC4022EJ-Hypnogram.edf",
    "SC4031E0-PSG.edf",
    "SC4031EC-Hypnogram.edf",
    "SC4032E0-PSG.edf",
    "SC4032EP-Hypnogram.edf",
    "SC4041E0-PSG.edf",
    "SC4041EC-Hypnogram.edf",
    "SC4042E0-PSG.edf",
    "SC4042EC-Hypnogram.edf",
    "SC4051E0-PSG.edf",
    "SC4051EC-Hypnogram.edf",
    "SC4052E0-PSG.edf",
    "SC4052EC-Hypnogram.edf",
    "SC4061E0-PSG.edf",
    "SC4061EC-Hypnogram.edf",
    "SC4062E0-PSG.edf",
    "SC4062EC-Hypnogram.edf",
    "SC4071E0-PSG.edf",
    "SC4071EC-Hypnogram.edf",
    "SC4072E0-PSG.edf",
    "SC4072EH-Hypnogram.edf",
    "SC4081E0-PSG.edf",
    "SC4081EC-Hypnogram.edf",
    "SC4082E0-PSG.edf",
    "SC4082EP-Hypnogram.edf",
    "SC4091E0-PSG.edf",
    "SC4091EC-Hypnogram.edf",
    "SC4092E0-PSG.edf",
    "SC4092EC-Hypnogram.edf",
    "SC4101E0-PSG.edf",
    "SC4101EC-Hypnogram.edf",
    "SC4102E0-PSG.edf",
    "SC4102EC-Hypnogram.edf",
    "SC4111E0-PSG.edf",
    "SC4111EC-Hypnogram.edf",
    "SC4112E0-PSG.edf",
    "SC4112EC-Hypnogram.edf",
    "SC4121E0-PSG.edf",
    "SC4121EC-Hypnogram.edf",
    "SC4122E0-PSG.edf",
    "SC4122EV-Hypnogram.edf",
    "SC4131E0-PSG.edf",
    "SC4131EC-Hypnogram.edf",
    "SC4141E0-PSG.edf",
    "SC4141EU-Hypnogram.edf",
    "SC4142E0-PSG.edf",
    "SC4142EU-Hypnogram.edf",
    "SC4151E0-PSG.edf",
    "SC4151EC-Hypnogram.edf",
    "SC4152E0-PSG.edf",
    "SC4152EC-Hypnogram.edf",
    "SC4161E0-PSG.edf",
    "SC4161EC-Hypnogram.edf",
    "SC4162E0-PSG.edf",
    "SC4162EC-Hypnogram.edf",
    "SC4171E0-PSG.edf",
    "SC4171EU-Hypnogram.edf",
    "SC4172E0-PSG.edf",
    "SC4172EC-Hypnogram.edf",
    "SC4181E0-PSG.edf",
    "SC4181EC-Hypnogram.edf",
    "SC4182E0-PSG.edf",
    "SC4182EC-Hypnogram.edf",
    "SC4191E0-PSG.edf",
    "SC4191EP-Hypnogram.edf",
    "SC4192E0-PSG.edf",
    "SC4192EV-Hypnogram.edf"
]


def main():
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in FILENAMES:
        file_url = BASE_URL + filename
        save_path = os.path.join(OUTPUT_DIR, filename)

        # if the file already exists and is not empty
        if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
            tqdm.write(f"File {filename} already exists. Lets skipp.")
            continue

        try:
            # stream the download to handle large files
            response = requests.get(file_url,  stream=True)
            response.raise_for_status()  # Raise error for inapropriate status codes

            # total file size for tqdm
            total_size = int(response.headers.get('content-length', 0))

            # open file and write chunks with progress bar
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename) as pbar:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        except requests.exceptions.RequestException as e:
            tqdm.write(f"I have an error downloading {filename}: {e}")

    print("\nAll downloads finished!")


if __name__ == "__main__":
    main()
