import os
import pandas as pd
import mne
from mne.io import read_raw_edf
import pyedflib
import numpy as np

# Paths
base_folder = r"F:\physionet.org\files\chbmit\1.0.0\chb02"
summary_file = os.path.join(base_folder, "summary_2.csv")
extraction_folder = os.path.join(base_folder, "extraction1")

# Create the extraction folder if it doesn't exist
os.makedirs(extraction_folder, exist_ok=True)

# Read the summary CSV file
summary_df = pd.read_csv(summary_file)

def save_edf(raw, output_file):
    """
    Save MNE Raw object as an EDF file using pyedflib.
    """
    n_channels = len(raw.ch_names)
    sfreq = int(raw.info["sfreq"])
    data = raw.get_data() * 1e6  # Convert data to microvolts
    channel_info = raw.info["chs"]

    # Create EDF writer
    edf_writer = pyedflib.EdfWriter(
        output_file,
        n_channels=n_channels,
        file_type=pyedflib.FILETYPE_EDF
    )

    # Define channel metadata
    channel_labels = raw.ch_names
    physical_min = np.min(data, axis=1)
    physical_max = np.max(data, axis=1)
    digital_min = -32768
    digital_max = 32767
    transducer = [""] * n_channels
    prefilter = [""] * n_channels

    # Set channel headers
    edf_writer.setSignalHeaders(
        [
            {
                "label": label,
                "dimension": "uV",
                "sample_frequency": sfreq,
                "physical_min": p_min,
                "physical_max": p_max,
                "digital_min": digital_min,
                "digital_max": digital_max,
                "transducer": trans,
                "prefilter": pre,
            }
            for label, p_min, p_max, trans, pre in zip(
                channel_labels, physical_min, physical_max, transducer, prefilter
            )
        ]
    )

    # Write data and close file
    edf_writer.writeSamples(data)
    edf_writer.close()


# Function to extract segments
def extract_segments(file_name, start_time, end_time, output_file):
    raw = read_raw_edf(os.path.join(base_folder, file_name), preload=True)
    sfreq = raw.info['sfreq']  # Sampling frequency
    start_sample = int(start_time * sfreq)
    end_sample = int(end_time * sfreq)
    extracted = raw.copy().crop(tmin=start_sample / sfreq, tmax=end_sample / sfreq)
    save_edf(extracted, os.path.join(extraction_folder, output_file))


# Process each row in the summary CSV
for index, row in summary_df.iterrows():
    file_name = row['File Name']
    seizure_start_time = row['Seizure Start Time (s)']
    seizure_end_time = row['Seizure End Time (s)']
    seizures = row['Seizures']
    
    # Create seizure and non-seizure files
    if seizures > 0:
        # Extract seizure segment
        extract_segments(file_name, seizure_start_time, seizure_end_time, f"seizure_{index + 1}.edf")

        # Extract non-seizure segments
        raw = read_raw_edf(os.path.join(base_folder, file_name), preload=True)
        sfreq = raw.info['sfreq']  # Sampling frequency
        total_duration = raw.times[-1]  # Total duration in seconds
        
        # Non-seizure before seizure start
        if seizure_start_time > 0:
            extract_segments(file_name, 0, seizure_start_time, f"non_{index + 1}_before.edf")
        
        # Non-seizure after seizure end
        if seizure_end_time < total_duration:
            extract_segments(file_name, seizure_end_time, total_duration, f"non_{index + 1}_after.edf")
    else:
        # Save the whole file as non-seizure
        raw = read_raw_edf(os.path.join(base_folder, file_name), preload=True)
        save_edf(raw, os.path.join(extraction_folder, f"non_{index + 1}.edf"))

print("Extraction process completed. Files saved in:", extraction_folder)
