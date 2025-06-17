import os
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from datasets import CWRU, Paderborn, Hust, UORED

def generate_spectrogram(dataset, metainfo, spectrogram_setup, signal_length, 
                         num_segments=None, output_dir=None, preprocessing="zscore"):
    dataset_name = dataset.__class__.__name__.lower()
    #dataset = eval(dataset_name + "()")

    if output_dir is None:
        output_dir = os.path.join("data/spectrograms", dataset_name.lower(), "")
    
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    if spectrogram_setup["noverlap"] >= spectrogram_setup["nperseg"]:
        raise ValueError("Error: `noverlap` must be less than `nperseg`.")

    for info in metainfo:
        basename = info["filename"]
        filepath = os.path.join('data/raw/', dataset_name.lower(), basename + '.mat')

        data, label = dataset.load_signal_by_path(filepath) #Load accelerometer data
        
        if preprocessing == "zscore": # Z-score normalization
            data = (data - np.mean(data)) / np.std(data)
        elif preprocessing == "rms":
            data = data / np.sqrt(np.mean(np.square(data)))
        elif preprocessing == "none":
            pass  
        
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
            Sxx_dB = 10 * np.log10(np.abs(Sxx) + 1e-8)  # Avoid log(0)
            #Sxx_dB = np.fliplr(abs(Sxx).T).T
            
            # Plot the spectrogram
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(Sxx_dB, cmap='jet', aspect='auto', origin='lower',
                           extent=[t.min(), t.max(), f.min(), f.max()], vmin=-100, vmax=0) #cmap='jet'
            ax.axis('off')

            # Save the spectrogram
            label_dir  = os.path.join(output_dir, label)
            os.makedirs(label_dir , exist_ok=True)  # Ensure directory exists

            output_path = os.path.join(label_dir , f"{basename}#{i+1}.png")
        
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            print(f"Spectrogram {output_path} - created.")
            plt.close(fig)
            

    print(f"Completed spectrogram generation for {dataset_name}.")