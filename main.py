from scripts.download_rawfile import download_rawfile
from scripts.spectrograms import generate_spectrogram
from scripts.copy_spectrogram_to_folds import copy_spectrogram_to_folds
from src.data_processing.dataset_manager import DatasetManager
from utils import load_yaml
from utils.dual_output import DualOutput  # Import the class from dual_output.py
from experimenter_vitclassifier_kfold import experimenter_classifier
from run_pretrain import experimenter
import sys
import pandas as pd
from datetime import datetime
import os

# Create logs directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Generate a timestamped log file name
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_filename = f"results/experiment_log_{timestamp}.txt"

# Redirect stdout
sys.stdout = DualOutput(log_filename)

csv_filename = "results/experiment_results.csv"

# Ensure CSV exists with headers
if not os.path.exists(csv_filename):
    pd.DataFrame(columns=[
        "model_type", "pretrain_model", "base_model", "num_classes",
        "num_epochs", "lr", "num_epochs_kf", "lr_kf", "batch_size", "rootdir",
        "first_datasets_name", "target_datasets_name", "perform_kfold",
        "mode","timestamp", "accuracy", "precision", "recall", "f1_score", "log_file","endtime"
    ]).to_csv(csv_filename, index=False)


# DOWNLOAD RAW FILES
def download():
    for dataset in ["CWRU", "UORED"]:
        download_rawfile(dataset)

# SPECTROGRAMS
def create_spectrograms():

    # Sets the number of segments
    num_segments = 20

    # Load the configuration files
    spectrogram_config = load_yaml('config/spectrogram_config.yaml')
    filter_config = load_yaml('config/filters_config.yaml')
    
    # Instantiate the data manager
        
    for dataset_name in spectrogram_config.keys():
        print(f"Starting the creation of the {dataset_name} spectrograms.")
        filter = filter_config[dataset_name]
        data_manager = DatasetManager(dataset_name)
        metainfo = data_manager.filter_data(filter)
        signal_length = spectrogram_config[dataset_name]["Split"]["signal_length"]
        spectrogram_setup = spectrogram_config[dataset_name]["Spectrogram"]
        
        # Creation of spectrograms    
        generate_spectrogram(metainfo, spectrogram_setup, signal_length, num_segments) 

# EXPERIMENTERS
def run_experimenter():
    #model = ResNet18() 
    model_type="DeiT"  # Options: "ViT", "DeiT", "DINOv2", "SwinV2", "CNN2D", "ResNet18"
    pretrain_model=True # pretrain or use saved 
    base_model=False # base model with no pre-train strategy nor use of weights saved
    perform_kfold=True
    
    experiment_params = {
        "model_type": model_type,
        "pretrain_model": pretrain_model,
        "base_model": base_model,
        "num_classes": 4,
        "num_epochs": 20,
        "lr": 0.00005,
        "num_epochs_kf": 16,
        "lr_kf": 0.00016,
        "batch_size": 32,
        "root_dir": "data/spectrograms",
        "first_datasets_name": ["CWRU"],
        "target_datasets_name": ["UORED"],
        "perform_kfold": perform_kfold,
        "mode": "supervised"  # "pretrain", "supervised", or "both"
    }
    
    metrics = experimenter_classifier(
        **experiment_params
    )

    # Extract metrics
    accuracy = metrics.get("accuracy", "N/A")
    precision = metrics.get("precision", "N/A")
    recall = metrics.get("recall", "N/A")
    f1_score = metrics.get("f1", "N/A")
    endtime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Append results to CSV
    result_data = {
        **experiment_params,
        "timestamp": timestamp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "log_file": log_filename,
        "endtime":endtime
    }
    
    df = pd.DataFrame([result_data])
    df.to_csv(csv_filename, mode='a', header=False, index=False)
    
    print("Experiment results saved to CSV.")


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f">> Start: {timestamp}")
    #download()
    #create_spectrograms()
    run_experimenter()
    
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f">> End: {timestamp}")
    # Close the log file
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # Reset stdout to the original