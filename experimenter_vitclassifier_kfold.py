import torch
import os
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.models import CNN2D, ViTClassifier, ResNet18, DeiTClassifier
from src.models.vitclassifier import train_and_save, load_trained_model
from scripts.evaluate_model_vitclassifier import kfold_cross_validation, resubstitution_test, one_fold_with_bias, one_fold_without_bias, evaluate_full_model

import sys
import logging
from utils.logginout import LoggerWriter
from utils.print_info import print_info

def compute_and_print_distribution(loader, class_to_idx, dataset_name):
    """
    Compute and print the distribution of classes in a DataLoader.
    Args:
        loader (DataLoader): The DataLoader to compute the distribution for.
        class_to_idx (dict): Mapping of class names to indices.
        dataset_name (str): The name of the dataset (e.g., UORED or CWRU).
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # Reverse mapping
    all_labels = []

    for _, labels in loader:
        all_labels.extend(labels.cpu().numpy())

    distribution = {idx_to_class[idx]: all_labels.count(idx) for idx in class_to_idx.values()}

    # Generate single-line print
    dist_str = " | ".join([f"{class_name}: {count}" for class_name, count in distribution.items()])
    print(f"Class Distribution for {dataset_name} | {dist_str}")

    return distribution

def enforce_consistent_mapping(datasets, desired_class_to_idx):
    """
    Ensures that all datasets have the same class-to-index mapping.
    Args:
        datasets (list): List of ImageFolder datasets.
        desired_class_to_idx (dict): The desired class-to-index mapping.
    """
    for dataset in datasets:
        dataset_class_to_idx = dataset.class_to_idx
        
        # Check if the dataset's mapping matches the desired mapping
        if dataset_class_to_idx != desired_class_to_idx:
            print(f"[warning] Dataset mapping {dataset_class_to_idx} does not match desired {desired_class_to_idx}. Remapping...")
            
            # Reorder the samples according to the desired mapping
            new_samples = []
            for path, label in dataset.samples:
                class_name = dataset.classes[label]  # Get class name from current label
                new_label = desired_class_to_idx[class_name]  # Map to desired label
                new_samples.append((path, new_label))
            
            # Update dataset's samples and class_to_idx
            dataset.samples = new_samples
            dataset.class_to_idx = desired_class_to_idx
            dataset.classes = list(desired_class_to_idx.keys())
    print("[info] Mappings enforced successfully.")

def experimenter_vitclassifier_kfold(use_vit=True, pretrain_model=False, base_model=True):
    
    # Toggle between use the pre-trained saved model or pre-train it
    # pretrain_model = True
    
    # Toggle between ViT and DeiT
    # use_vit = False  # Set to True for ViT, False for DeiT

    model_class = ViTClassifier if use_vit else DeiTClassifier
    model = model_class(num_classes=4).to("cuda")
    
    # Training parameters 
    num_epochs_vit_train = 10
    lr_vit_train = 0.00005
    batch_size = 32
    
    # Class-to-index mapping
    class_to_idx = {'B': 0, 'I': 1, 'N': 2, 'O': 3}
    
    # Load datasets
    train_datasets_name = ["CWRU"]
    test_datasets_name = ["UORED"]

    root_dir = "data/spectrograms"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
     
    # Load train and test datasets
    train_datasets = [ImageFolder(os.path.join(root_dir, ds.lower()), transform) for ds in train_datasets_name]
    test_datasets = [ImageFolder(os.path.join(root_dir, ds.lower()), transform) for ds in test_datasets_name]

    # Enforce consistent mapping
    enforce_consistent_mapping(train_datasets, class_to_idx)
    enforce_consistent_mapping(test_datasets, class_to_idx)

    # Combine datasets if necessary
    pretrain_concated_dataset = ConcatDataset(train_datasets)
    test_concated_dataset = ConcatDataset(test_datasets)
 
    train_idx, eval_idx = train_test_split(
        range(len(pretrain_concated_dataset)), 
        test_size=0.2, 
        stratify=[y for _, y in pretrain_concated_dataset]
    )

    # Define loaders
    pretrain_train_loader = DataLoader(
        torch.utils.data.Subset(pretrain_concated_dataset, train_idx), 
        batch_size=batch_size, shuffle=True, num_workers=4
    )

    pretrain_eval_loader = DataLoader(
        torch.utils.data.Subset(pretrain_concated_dataset, eval_idx), 
        batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    pretrain_loader = DataLoader(pretrain_concated_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_concated_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Compute and print distributions using dataset names
    for dataset_name, dataset_loader in zip(train_datasets_name, [pretrain_loader]):
        print(f"\n>> Calculating distribution for pre-train dataset ({dataset_name})...")
        train_distribution = compute_and_print_distribution(dataset_loader, class_to_idx, dataset_name)

    for dataset_name, dataset_loader in zip(test_datasets_name, [test_loader]):
        print(f"\n>> Calculating distribution for test dataset ({dataset_name})...")
        test_distribution = compute_and_print_distribution(dataset_loader, class_to_idx, dataset_name)
    
    # Save path and experiment log
    saved_model_path = "saved_models/vit_classifier.pth" if use_vit else "saved_models/deit_classifier.pth"
    title = f"Transfer Learning: Addressing cross Datasets with {'ViT' if use_vit else 'DeiT'}Classifier"
    print_info("Experiment", [title])
    if not base_model:
        print(f"Saved model path: {saved_model_path}")
        
    # Print the class-to-index mapping with dataset names
    for dataset_name, dataset in zip(train_datasets_name, train_datasets):
        print(f"Pre-train dataset ({dataset_name}) mapping: {dataset.class_to_idx}")

    for dataset_name, dataset in zip(test_datasets_name, test_datasets):
        print(f"Test dataset ({dataset_name}) mapping: {dataset.class_to_idx}")
    
    # Instantiate the ViTClassifier or DeiTClassifier and train it with train_loader to narrow the model context
    if not base_model:
        if pretrain_model: 
            model = model_class().to("cuda")
            print("Pre-training according request.")
            train_and_save(model, pretrain_train_loader, pretrain_eval_loader, num_epochs_vit_train, lr_vit_train, saved_model_path)
        else:
            print("No Pre-training started, using a pre-train saved file.")
            # Load the trained model for testing/evaluation
            model = load_trained_model(model_class, saved_model_path, num_classes=len(class_to_idx)).to("cuda")
    else:
        initial_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(copy.deepcopy(initial_state))
        print(f"Using raw base model {'ViT' if use_vit else 'DeiT'}Classifier with no pre-train.")
    
    # Running the experiment 
    num_epochs = 10
    lr = 0.00005
    #group_by = "rpm" 
    #group_by = "extent_damage"
    #group_by = "condition_bearing_health"
    #group_by = "damage_method"
    group_by = ""

    # Evaluating the model
    
    # resubstitution_test(
    #     model,
    #     ds_test_loader,
    #     num_epochs,
    #     lr,
    #     class_names = list(class_to_idx.keys())
    # )
    
    # one_fold_with_bias(
    #     model,
    #     ds_test_loader,
    #     num_epochs,
    #     lr,
    #     class_names = list(class_to_idx.keys())
    # )
    
    # one_fold_without_bias(
    #     model,
    #     ds_test_loader,
    #     num_epochs,
    #     lr,
    #     class_names = list(class_to_idx.keys())
    # )
    
    kfold_cross_validation(
        model, 
        test_loader, 
        num_epochs, 
        lr, 
        group_by, 
        class_names = list(class_to_idx.keys()), 
        n_splits=4)

def run_experimenter():     
    experimenter_vitclassifier_kfold()

if __name__ == "__main__":
    #sys.stdout = LoggerWriter(logging.info, "kfold-vitclassifier")
    run_experimenter()
    
    
