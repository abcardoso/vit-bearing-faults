import os
import sys
import pandas as pd
from datetime import datetime
from utils.dual_output import DualOutput
from contextlib import redirect_stdout
from main import create_spectrograms, run_experimenter  # Import functions from main.py

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Initialize Experiment Batch Logging
timestamp1 = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# log_filename1 = f"results/experiment_log_batch_{timestamp1}.txt"
print(f">> Start Experiment Batch: {timestamp1}")

# Define parameter values
model_types = ["ViT", "DeiT", "DINOv2", "SwinV2", "MAE", "CNN2D"]
# preprocessing_methods = ["rms","zscore"]
preprocessing_methods = ["none"]
train_test_tuples = [
    (["1", "3", "5", "7"], "9"),
    (["1", "3", "5", "9"], "7"),
    (["1", "3", "7", "9"], "5"),
    (["1", "5", "7", "9"], "3"),
    (["3", "5", "7", "9"], "1"),
    (["2", "4", "6", "8"], "10"),
    (["2", "4", "6", "10"], "8"),
    (["2", "4", "8", "10"], "6"),
    (["2", "6", "8", "10"], "4"),
    (["4", "6", "8", "10"], "2"),
]

# Experiment Execution Loop
for train_domains, test_domain in train_test_tuples:
    for preprocessing in preprocessing_methods:
        # Create spectrograms only once per train/test combination
        print(f"📢 Creating spectrograms for Train: {train_domains} | Test: {test_domain} | Preprocessing: {preprocessing}")
        create_spectrograms(use_domain_split=True, train_domains=train_domains, test_domain=test_domain, preprocessing=preprocessing)

        for model_type in model_types:
            print(f"\n🚀 Running Experiment: Model={model_type}, Preprocessing={preprocessing}, Train={train_domains}, Test={test_domain}")

            run_experimenter(
                use_domain_split=True, train_domains=train_domains, test_domain=test_domain,
                preprocessing=preprocessing, model_type=model_type,
                pretrain_model=False, base_model=True, perform_kfold=True
            )

# End Experiment Batch
timestamp2 = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print(f">> End Experiment Batch: {timestamp2}")
