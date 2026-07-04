import os
import mne
import moabb
from braindecode.datasets import MOABBDataset

# ================================
# 1. DATA LOADING
# ================================

# Saving data in SCRATCH to avoid exceeding disk quota 
data_path = '/scratch/fsapere/TinyEEG/data'

# Create folder if it is not existing already
os.makedirs(data_path, exist_ok=True)

# Configure MNE and MOABB to use the path in scratch
mne.set_config('MNE_DATA', data_path)
moabb.set_download_dir(data_path)

print(f"Download folder: {data_path}")
print("Downloading BNCI2014_001 dataset...")

# Use braindecode to install the data set, using subject_ids=None to donwload all 
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=1)
# change subject id later to have a more robust training, for now I am just seeing if it does work

print(f"Download completed. Dataset is @: {data_path}")

# ================================
# 2. DATA PRE-PROCESSING & EPOCH CREATION
# ================================

from braindecode.preprocessing import (
    preprocess, Preprocessor, exponential_moving_standardize,
    create_windows_from_events
)

print("Starting preprocessing (Filters, Downsampling, Standardization)...")
preprocessors = [
    # Select only EEG channels
    Preprocessor('pick_types', eeg=True, meg=False, stim=False, verbose=False),
    # Scale from Volts (MNE default) to microVolts
    Preprocessor(lambda x: x * 1e6),
    # Band-pass filter from 4.0 Hz to 38.0 Hz
    Preprocessor('filter', l_freq=4.0, h_freq=38.0, verbose=False),
    # Downsampling to 128 Hz
    Preprocessor('resample', sfreq=128, verbose=False),
    # Continuous standardization
    Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=1000)
]

preprocess(dataset, preprocessors)
print("Preprocessing completed.")

print("Creating windows from events (trial-based)...")
sfreq = 128
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=int(-0.5 * sfreq),
    trial_stop_offset_samples=0,
    preload=True
)
print(f"Total windows created: {len(windows_dataset)}")

# ================================
# 3. VERIFICATION
# ================================
print("\nVerifying DataLoader output...")
import torch
from torch.utils.data import DataLoader

# Create a DataLoader to extract one batch
dataloader = DataLoader(windows_dataset, batch_size=32, shuffle=True)
batch_X, batch_y, batch_ind = next(iter(dataloader))

print(f"X tensor shape (batch_size, channels, time): {batch_X.shape}")
print(f"Unique targets in batch: {torch.unique(batch_y).tolist()}")
print("Data Loader verified successfully!") 