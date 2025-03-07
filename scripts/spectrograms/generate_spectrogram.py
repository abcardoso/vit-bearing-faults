import os
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from datasets import CWRU, Paderborn, Hust, UORED

def generate_spectrogram(dataset, metainfo, spectrogram_setup, signal_length, num_segments=None):
    dataset_name = dataset.__class__.__name__.lower()
    #dataset = eval(dataset_name + "()")

    domain_folder = dataset.get_domain_folder("")

    for info in metainfo:
        basename = info["filename"]
        filepath = os.path.join('data/raw/', dataset_name.lower(), basename + '.mat')

        # Load signal and label
        data, label = dataset.load_signal_by_path(filepath)

        # Normalize and detrend the signal
        data = (data - np.mean(data)) / np.std(data)  # Z-score normalization
        detrended_data = signal.detrend(data)

        # Determine the number of segments
        total_samples = detrended_data.shape[0]
        n_segments = total_samples // signal_length
        n_max_segments = min([num_segments or n_segments, n_segments])

        for i in range(n_max_segments):
            start_idx = i * signal_length
            end_idx = start_idx + signal_length
            segment = detrended_data[start_idx:end_idx]

            # Compute STFT (short-time Fourier transform)
            f, t, Sxx = signal.stft(segment, **spectrogram_setup)

            # Convert to decibels for better scaling
            Sxx_dB = 10 * np.log10(np.abs(Sxx[:382, :]**2) + 1e-8)  # Add epsilon to avoid log(0)

            # Plot the spectrogram
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(Sxx_dB, cmap='jet', aspect='auto', origin='lower',
                           extent=[t.min(), t.max(), f.min(), f.max()])
            ax.axis('off')

            # Save the spectrogram
            output_dir = os.path.join('data/spectrograms', dataset_name.lower(), domain_folder, label)
            os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

            output_path = os.path.join(output_dir, f"{basename}#{i+1}.png")
        
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            print(f"Spectrogram {output_path} - created.")
            plt.close(fig)
            

    print(f"Completed spectrogram generation for {dataset_name}.")