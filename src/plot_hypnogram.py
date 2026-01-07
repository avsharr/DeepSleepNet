import os
import mne
import matplotlib.pyplot as plt

# get the path of the data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data", "raw")

# just an example of the annotated data
psg_file = os.path.join(data_dir, "SC4001E0-PSG.edf")
hyp_file = os.path.join(data_dir, "SC4001EC-Hypnogram.edf")

# read files
raw = mne.io.read_raw_edf(psg_file, preload=False)  # signal
annot = mne.read_annotations(hyp_file)             # hypnogram annotations

# merge annotations with the raw signal
raw.set_annotations(annot, emit_warning=False)

# map text labels  to integers for plotting
event_id = {
    'Sleep stage W': 1,
    'Sleep stage 1': 2,
    'Sleep stage 2': 3,
    'Sleep stage 3': 4,
    'Sleep stage 4': 4, # merge stages 3 and 4 into deep sleep (N3) as sleep annotation rules were changed
    'Sleep stage R': 5
}

# extract events from annotations
events, _ = mne.events_from_annotations(
    raw,
    event_id=event_id,
    chunk_duration=30.0 # standard sleep epoch is 30 seconds
)

# plot the Hypnogram
fig = mne.viz.plot_events(
    events,
    sfreq=raw.info['sfreq'],
    first_samp=raw.first_samp,
    event_id=event_id,
    show=False
)

# title and labels
plt.title("Hypnogram (Sleep Stages)")
plt.ylabel("Stage")
plt.xlabel("Time (samples)")
plt.grid(True, alpha=0.3)
plt.show(block=True)