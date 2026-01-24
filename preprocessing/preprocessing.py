import os
import numpy as np
import mne

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA = os.path.join(PROJECT_ROOT, 'data', 'preprocessed')
os.makedirs(PROCESSED_DATA, exist_ok=True)

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
    labels = np.asarray(labels).flatten()
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    return np.array([total / (num_classes * counts[i]) if counts[i] > 0 else 1.0 for i in range(num_classes)])


def process(psg_file, hyp_file, save):
    raw = mne.io.read_raw_edf(psg_file, preload=True)
    raw.set_annotations(mne.read_annotations(hyp_file))

    onsets = raw.annotations.onset
    descriptions = raw.annotations.description
    sleep_stages = ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R']
    sleep_indices = [i for i, d in enumerate(descriptions) if d in sleep_stages]

    if not sleep_indices:
        return

    first_sleep = onsets[sleep_indices[0]]
    last_sleep = onsets[sleep_indices[-1]]
    t_start = max(0, first_sleep - 1800)
    t_end = min(raw.times[-1], last_sleep + 1800)
    raw.crop(tmin=t_start, tmax=t_end)

    if 'EEG Fpz-Cz' not in raw.ch_names:
        return
    raw.pick_channels(['EEG Fpz-Cz'])

    event, event_id = mne.events_from_annotations(raw, event_id=MAPPING, chunk_duration=30.0, verbose=False)
    epochs = mne.Epochs(raw, event, event_id, tmin=0, tmax=30.0 - 1.0/raw.info['sfreq'], baseline=None, preload=True, verbose=False)
    X = epochs.get_data() * 1e6
    y = epochs.events[:, 2]
    np.savez_compressed(save, x=X, y=y)
