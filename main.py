from scripts.download_rawfile import download_rawfile
from scripts.spectrograms import generate_spectrogram
from scripts.copy_spectrogram_to_folds import copy_spectrogram_to_folds
from src.data_processing.dataset_manager import DatasetManager
from utils import load_yaml
from utils.dual_output import DualOutput  # Import the class from dual_output.py
from experimenter_vitclassifier_kfold import experimenter_classifier_v2
from run_pretrain import experimenter
from datasets.uored import UORED
from datasets.cwru import CWRU
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
        "dataset_name", "perform_kfold",
        "mode", "use_domain_split", "train_domains", "test_domain",
        "start", "endtime", "time_spent",  # Added time_spent column
        "accuracy", "precision", "recall", "f1_score",
        "train_accuracy", "validation_accuracy", "test_accuracy",
        "log_file", "preprocessing"
    ]).to_csv(csv_filename, index=False)


# DOWNLOAD RAW FILES
def download(dataset_name):
    #for dataset in ["CWRU", "UORED"]:
    download_rawfile(dataset_name)


def get_metainfo(dataset, domain_files):
    metainfo = []
    
    if hasattr(dataset, "semantic_to_file"):
        # === CWRU logic ===
        for semantic in domain_files:
            if semantic.startswith("C"):
                continue  # Skip acoustic

            file_id = dataset.semantic_to_file.get(semantic)
            if file_id:
                annotation = next((x for x in dataset.annotation_file if x["filename"] == file_id), None)
                if annotation:
                    metainfo.append({
                        "filename": file_id,
                        "label": annotation["label"]
                    })
                else:
                    print(f"[WARNING] Annotation missing for file_id {file_id}")
            else:
                print(f"[WARNING] Mapping missing for semantic name {semantic}")
    else:
        # === UORED logic ===
        for semantic in domain_files:
            if semantic.startswith("C"):
                continue
            file_id = semantic.replace("-", "_")  # Normalize filename
            label_code = semantic.split("-")[0]
            metainfo.append({"filename": file_id, "label": label_code})

    return metainfo

# SPECTROGRAMS
def create_spectrograms(use_domain_split=False, train_domains=None, test_domain=None, 
                        preprocessing="none", target_dataset=None, num_segments=20):

    # Sets the number of segments
    #num_segments = 20

    # Load the configuration files
    spectrogram_config = load_yaml('config/spectrogram_config.yaml')
    filter_config = load_yaml('config/filters_config.yaml')
    
    dataset_names = (
        [target_dataset] if target_dataset else spectrogram_config.keys()
    )
        
    for dataset_name in dataset_names:
        print(f"Starting the creation of the {dataset_name} spectrograms.")
        filter = filter_config[dataset_name]

        if dataset_name == "UORED":
            dataset = UORED(use_domain_split=use_domain_split, train_domains=train_domains, test_domain=test_domain)
        elif dataset_name == "CWRU":
            dataset = CWRU(use_domain_split=use_domain_split, train_domains=train_domains, test_domain=test_domain)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_output_dir = os.path.join("data/spectrograms", dataset_name.lower())
        os.makedirs(dataset_output_dir, exist_ok=True)

        signal_length = spectrogram_config[dataset_name]["Split"]["signal_length"]
        spectrogram_setup = spectrogram_config[dataset_name]["Spectrogram"]
        
        if use_domain_split:
            train_output_dir = os.path.join("data/spectrograms", dataset_name.lower(), f"train_domains_{'_'.join(train_domains)}")
            os.makedirs(train_output_dir, exist_ok=True)
            
            for train_domain in train_domains:
                print(f"Generating spectrograms for Train/Validation domain: {train_domain}")
                dataset.train_domains = [train_domain]
                domain_train_files = dataset.domain_mapping[train_domain]

                metainfo = get_metainfo(dataset, domain_train_files)

                print(f"[DEBUG] Domain: {train_domain} → {len(metainfo)} files: {[m['filename'] for m in metainfo]}")
                generate_spectrogram(
                    dataset, metainfo, spectrogram_setup, signal_length, num_segments, 
                    output_dir=train_output_dir, preprocessing=preprocessing
                    )

            # Generate spectrograms for the test domain separately
            test_output_dir = os.path.join("data/spectrograms", dataset_name.lower(), f"test_domain_{test_domain}")
            print(f"Generating spectrograms for Test domain: {test_domain}")
            dataset.test_domain = test_domain
            
            domain_test_files = dataset.domain_mapping[test_domain]
            
            metainfo = get_metainfo(dataset, domain_test_files)

            print(f"[DEBUG] Domain: {test_domain} → {len(metainfo)} files: {[m['filename'] for m in metainfo]}")
            
            generate_spectrogram(
                dataset, metainfo, spectrogram_setup, signal_length, 
                num_segments, output_dir=test_output_dir, preprocessing=preprocessing 
                )
        
        else:
            print(f"Generating spectrograms for {dataset_name} (No domain split).")
            data_manager = DatasetManager(dataset_name)
            filter = filter_config[dataset_name]
            metainfo = data_manager.filter_data(filter)
            
            generate_spectrogram(dataset, metainfo, spectrogram_setup, signal_length, num_segments,preprocessing=preprocessing)
      
        print(f"Completed spectrogram generation for {dataset_name}; signal_length: {signal_length} ; spectrogram_setup: {spectrogram_setup}; preprocessing {preprocessing}. Directory: {dataset_output_dir} ")
        
                
# EXPERIMENTERS
def run_experimenter(use_domain_split=False, train_domains=None, test_domain=None, preprocessing="none",
                     model_type="CNN2D", pretrain_model=False, base_model=True, perform_kfold=True, dataset_name="CWRU"):
    
    
    start_time = datetime.now()
        
    experiment_params = {
        "model_type": model_type,
        "pretrain_model": pretrain_model,
        "base_model": base_model,
        "num_classes": 4,
        "num_epochs": 20,
        "lr": 0.00005,
        "num_epochs_kf": 30,
        "lr_kf": 0.00005,
        "batch_size": 32,
        "root_dir": "data/spectrograms",
        "dataset_name": dataset_name,
        "perform_kfold": perform_kfold,
        "mode": "supervised",  # "pretrain", "supervised", or "both"
        "use_domain_split": use_domain_split,
        "train_domains": train_domains,
        "test_domain": test_domain
    }
       
    metrics = experimenter_classifier_v2(
        **experiment_params
    )

    # Extract metrics
    accuracy = metrics.get("accuracy", "N/A")
    precision = metrics.get("precision", "N/A")
    recall = metrics.get("recall", "N/A")
    f1_score = metrics.get("f1", "N/A")
    train_accuracy = metrics.get("train_accuracy","N/A")
    validation_accuracy = metrics.get("validation_accuracy","N/A")
    test_accuracy = metrics.get("test_accuracy","N/A")
    
    end_time = datetime.now()
    time_spent = (end_time - start_time).total_seconds()  # Compute time spent in seconds

    # Append results to CSV
    result_data = {
        **experiment_params,
        "start": start_time.strftime("%Y-%m-%d-%H-%M-%S"),
        "endtime": end_time.strftime("%Y-%m-%d-%H-%M-%S"),
        "time_spent": time_spent,  # New column
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "log_file": log_filename,
        "preprocessing":preprocessing
    }
    
    df = pd.DataFrame([result_data])
    df.to_csv(csv_filename, mode='a', header=False, index=False)
    
    print("Experiment results saved to CSV.")


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f">> Start: {timestamp}")
    
    use_domain_split = True  # Toggle domain-based splitting
    train_domains=["1", "2", "3", "4", "5", "6", "7", "8"]  # Multiple domains for Train/Validation - from 1 to 10 based on Sehri et al.
    test_domain="9"
    preprocessing = "rms" # "zscore" (Standardization) "rms" (Root Mean Square) "none"
    model_type="DeiT"  # Options: "ViT", "DeiT", "DINOv2", "SwinV2", "MAE","CNN2D", "ResNet18"
    pretrain_model=False # pretrain or use saved 
    base_model=True # base model with no pre-train strategy neither use of weights saved
    perform_kfold=True
    create_sp = True
    download_raw = True
    dataset_name = "CWRU"
    num_segments = 20
    
    if download_raw:
        download(dataset_name)
    if create_sp:
        create_spectrograms(use_domain_split, train_domains, test_domain, preprocessing, dataset_name, num_segments) 
    
    run_experimenter(use_domain_split, train_domains, test_domain, preprocessing,
                     model_type, pretrain_model, base_model, perform_kfold, dataset_name)
    
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f">> End: {timestamp}")

    # Close the log file
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # Reset stdout to the original