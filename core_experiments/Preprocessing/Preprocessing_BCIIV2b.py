import os
import numpy as np
import mne
from scipy.stats import zscore

dir = 'BCICIV_2b'
os.makedirs(dir, exist_ok=True)
# subjs = [1, 2, 3]
subjs = list(range(1, 10))
Sessions = [1, 2]


for sub in subjs:

    for session in Sessions:
        # Load GDF file
        raw = mne.io.read_raw_gdf('BCICIV_2b_gdf/B0{0}0{1}T.gdf'.format(sub, session), preload=True)


        # Extract events (focus on Left=9, Right=10)
        events, events_id = mne.events_from_annotations(raw)
        events_2class = events[np.isin(events[:, 2], [events_id['769'], events_id['770']])] 

        # epochs = mne.Epochs(raw, 
        #             events=events_2class,
        #             event_id={'Left': events_id['769'], 'Right': events_id['770']},
        #             tmin=0.5,   # Start 0.5s after cue (MI onset)
        #             tmax=2.5,   # End 2.5s after cue (2s duration)
        #             baseline=None,  # No baseline correction
        #             preload=True)
        
        epochs = mne.Epochs(raw, 
                    events=events_2class,
                    event_id={'Left': events_id['769'], 'Right': events_id['770']},
                    tmin= -2,   # 1s after fixation = 2s before cue
                    tmax= 4,   # Ends at 7s = 4s after cue
                    baseline=None,  # No baseline correction
                    preload=True)
        
        epochs.pick_channels(['EEG:C3', 'EEG:Cz', 'EEG:C4'])  # Motor cortex channels
        X = epochs.get_data()  # Shape: (n_trials, 3_channels, 500_timepoints) 
        X = zscore(X, axis = 2)
        if sub == 1 and session == 2:
            y = epochs.events[:, 2] - 4 
            print(y)
            np.save('BCICIV_2b/Subj{0}_{1}_y.npy'.format(sub, session), y)

        else:
            y = epochs.events[:, 2] - 10
            np.save('BCICIV_2b/Subj{0}_{1}_y.npy'.format(sub, session), y)

        np.save('BCICIV_2b/Subj{0}_{1}_X.npy'.format(sub, session), X[:,:,:1500])
