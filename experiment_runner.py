import os
import sys
import pandas as pd
from datetime import datetime
from utils.dual_output import DualOutput
from contextlib import redirect_stdout
from collections import defaultdict
from main import create_spectrograms, run_experimenter  # Import functions from main.py

# Ensure results directory exists ...
os.makedirs("results", exist_ok=True)

# Initialize Experiment Batch Logging
timestamp1 = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# log_filename1 = f"results/experiment_log_batch_{timestamp1}.txt"
print(f">> Start Experiment Batch: {timestamp1}")

# Define parameter values
model_types = ["ViT", "DeiT", "DINOv2", "SwinV2", "MAE", "CNN2D"]
# model_types = ["ViT","DINOv2"]
preprocessing_methods = ["none","zscore", "rms"]
#preprocessing_methods = ["zscore", "rms"]
# preprocessing_methods = ["rms"]
dataset_name = "CWRU"
num_segments = 20
train_test_tuples = [
    # (["1", "2", "3", "4", "5", "6", "7", "8"], "9"),
    # (["1", "2", "3", "4", "5", "6", "7", "8"], "10"),
    # (["1", "2", "3", "4", "5", "6", "7", "8"], "11"),
    # (["1", "2", "3", "4", "5", "6", "7", "8"], "12"),
    # (["5", "6", "7", "8", "9", "10", "11", "12"], "1"),
    # (["5", "6", "7", "8", "9", "10", "11", "12"], "2"),
    # (["5", "6", "7", "8", "9", "10", "11", "12"], "3"),
    # (["5", "6", "7", "8", "9", "10", "11", "12"], "4"),
     (["1", "2", "3", "4", "9", "10", "11", "12"], "5"),
     (["1", "2", "3", "4", "9", "10", "11", "12"], "6"),
     (["1", "2", "3", "4", "9", "10", "11", "12"], "7"),
     (["1", "2", "3", "4", "9", "10", "11", "12"], "8"),
]

#Experiment Execution Loop
for train_domains, test_domain in train_test_tuples:
    for preprocessing in preprocessing_methods:
        # Create spectrograms only once per train/test combination
        print(f"📢 Creating spectrograms for Train: {train_domains} | Test: {test_domain} | Preprocessing: {preprocessing}")
        create_spectrograms(use_domain_split=True, train_domains=train_domains, 
                            test_domain=test_domain, preprocessing=preprocessing, target_dataset=dataset_name, num_segments=num_segments)

        for model_type in model_types:
            print(f"\n🚀 Running Experiment: Model={model_type}, Preprocessing={preprocessing}, Train={train_domains}, Test={test_domain}")

            run_experimenter(
                use_domain_split=True, train_domains=train_domains, test_domain=test_domain,
                preprocessing=preprocessing, model_type=model_type,
                pretrain_model=False, base_model=True, perform_kfold=True, dataset_name=dataset_name
            )
# base_path = 'data/spectrograms/cwru'
# print(os.listdir('data/spectrograms/cwru/test_domain_1'))

# def count_images(folder_prefix):
#     counts = defaultdict(int)
#     for folder in os.listdir(base_path):
#         if folder.startswith(folder_prefix):
#             full_path = os.path.join(base_path, folder)
#             for root, _, files in os.walk(full_path):
#                 counts[folder] += len([f for f in files if f.endswith('.png')])
#     return counts

# train_counts = count_images('train_domains')
# test_counts = count_images('test_domain')

# print("Training image counts:")
# for k, v in train_counts.items():
#     print(f"{k}: {v} images")

# print("\nTesting image counts:")
# for k, v in test_counts.items():
#     print(f"{k}: {v} images")

# End Experiment Batch
timestamp2 = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print(f">> End Experiment Batch: {timestamp2}")
