import os
import argparse
import numpy as np
import mne
from scipy.stats import zscore

def process_bci_data(data_dir: str, output_dir: str):
    """
    Preprocess BCI Competition IV Dataset 2b GDF files and save preprocessed 
    arrays (.npy) to output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    subjs = list(range(1, 10))
    sessions = [1, 2]

    for sub in subjs:
        for session in sessions:
            gdf_file = os.path.join(data_dir, f'B0{sub}0{session}T.gdf')
            
            if not os.path.exists(gdf_file):
                print(f"Skipping: {gdf_file} (File not found)")
                continue

            print(f"Processing Subject {sub}, Session {session}...")
            
            # Load raw data suppressing MNE's verbose output unless needed
            raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)

            # Extract events
            events, events_id = mne.events_from_annotations(raw, verbose=False)
            
            # Left Hand = 769, Right Hand = 770
            target_ids = [events_id['769'], events_id['770']]
            events_2class = events[np.isin(events[:, 2], target_ids)]

            epochs = mne.Epochs(
                raw, 
                events=events_2class,
                event_id={'Left': events_id['769'], 'Right': events_id['770']},
                tmin=-2.0,   # 2s before cue
                tmax=4.0,    # 4s after cue
                baseline=None,
                preload=True,
                verbose=False
            )
            
            epochs.pick_channels(['EEG:C3', 'EEG:Cz', 'EEG:C4'])
            X = epochs.get_data()  # Shape: (n_trials, 3_channels, n_samples)
            X = zscore(X, axis=2)

            # Standardize label encoding (0 for Left, 1 for Right)
            y = (epochs.events[:, 2] == events_id['770']).astype(int)

            # Save arrays
            np.save(os.path.join(output_dir, f'Subj{sub}_{session}_y.npy'), y)
            np.save(os.path.join(output_dir, f'Subj{sub}_{session}_X.npy'), X[:, :, :1500])

    print("Processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess BCI Competition IV 2b GDF dataset.")
    parser.add_argument(
        '--data-dir', 
        type=str, 
        required=True, 
        help="Path to directory containing input GDF files (e.g., ./BCICIV_2b_gdf)"
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./BCICIV_2b', 
        help="Directory where output .npy files will be saved (default: ./BCICIV_2b)"
    )
    
    args = parser.parse_args()
    process_bci_data(args.data_dir, args.output_dir)