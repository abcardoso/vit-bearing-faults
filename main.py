from scripts.download_rawfile import download_rawfile
from scripts.spectrograms import generate_spectrogram
from scripts.copy_spectrogram_to_folds import copy_spectrogram_to_folds
from src.data_processing.dataset_manager import DatasetManager
from utils import load_yaml
from utils.dual_output import DualOutput  # Import the class from dual_output.py
from experimenter_vitclassifier_kfold import experimenter_classifier_kfold
from run_pretrain import experimenter
import sys
from datetime import datetime
import os

# Create logs directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Generate a timestamped log file name
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_filename = f"results/experiment_log_{timestamp}.txt"

# Redirect stdout
sys.stdout = DualOutput(log_filename)


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
    model_type=""  # Options: "ViT", "DeiT", "DINOv2", "SwinV2", "CNN2D"
    pretrain_model=True # pretrain or use saved 
    base_model=False # base model with no pre-train strategy nor use of weights saved
    perform_kfold=True
    
    experimenter_classifier_kfold(
        model_type=model_type,
        pretrain_model=pretrain_model,
        base_model=base_model,
        num_classes=4,
        num_epochs=20,
        lr=0.00005,
        num_epochs_kf=10,
        lr_kf=0.00005,
        batch_size=32,
        root_dir="data/spectrograms",
        train_datasets_name=["CWRU"],
        test_datasets_name=["UORED"],
        perform_kfold=perform_kfold
    )


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