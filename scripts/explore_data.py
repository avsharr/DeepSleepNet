import os
import matplotlib.pyplot as plt
import mne

# project root (parent of scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(ROOT, "data", "raw")

# just an example of a data
psg_file = os.path.join(data_folder, "SC4001E0-PSG.edf")

raw = mne.io.read_raw_edf(psg_file, preload=True)

print(raw.info)

raw.plot(duration=30, n_channels=5, scalings='auto', title="First 30 seconds of sleep")
plt.show()
