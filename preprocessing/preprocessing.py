import os
import numpy as np
import mne

# project root (parent of preprocessing/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA = os.path.join(PROJECT_ROOT, 'data', 'preprocessed')

# check if we have our directory
os.makedirs(PROCESSED_DATA, exist_ok=True)

# map stages from experts with numbers (due to new AASM standard)
MAPPING = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

CLASS_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
NUM_CLASSES = 5


def compute_class_weights(labels, num_classes=5):
    """
    Compute class weights using inverse frequency (balanced).
    weight[i] = n_samples / (n_classes * count_of_class_i)
    """
    labels = np.asarray(labels).flatten()
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        if counts[i] > 0:
            weights.append(total / (num_classes * counts[i]))
        else:
            weights.append(1.0)
    return np.array(weights)


def process(psg_file, hyp_file, save):
    raw = mne.io.read_raw_edf(psg_file, preload=True)
    annotations = mne.read_annotations(hyp_file)
    raw.set_annotations(annotations)  # set appropriate annotations for each data

    # crop the data to prevent the class unbalance (30 minutes before and 30 minutes after sleep)
    onsets = annotations.onset
    descriptions = annotations.description

    # indices for all stages except awake stage
    sleep_stages = ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R']
    sleep_indices = [i for i, d in enumerate(descriptions) if d in sleep_stages]

    # actual time of falling asleep and waking up
    first_sleep_time = onsets[sleep_indices[0]]
    last_sleep_time = onsets[sleep_indices[-1]]

    # include 30 minutes (1800 sec) before and after sleep
    t_start = max(0, first_sleep_time - 1800)  # take the beginning of the recording if a person fall asleep in less than 30 minutes
    t_end = min(raw.times[-1], last_sleep_time + 1800)  # take the end of the recording is person woke up less than 30 min before the end

    raw.crop(tmin=t_start, tmax=t_end)

    # now lets choose only one EEG channel (Fpz-Cz)
    if 'EEG Fpz-Cz' not in raw.ch_names:
        print(f"Warning: EEG Fpz-Cz not found in {raw.ch_names}, skipping.")
        return
    raw.pick_channels(['EEG Fpz-Cz'])

    event, event_id = mne.events_from_annotations(raw, event_id=MAPPING, chunk_duration=30.0, verbose=False)
    epochs = mne.Epochs(raw, event, event_id, tmin=0, tmax=30.0 - 1.0/raw.info['sfreq'], baseline=None, preload=True, verbose=False)
    X = epochs.get_data() * 1e6
    y = epochs.events[:, 2]
    np.savez_compressed(save, x=X, y=y)
