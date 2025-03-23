import numpy as np
import copy
import torch
import cv2
import sys
import os
from datasets import UORED, CWRU
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
from torchvision.datasets import ImageFolder
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from src.models import CNN2D, ViTClassifier, ResNet18, DeiTClassifier, DINOv2WithRegistersClassifier, SwinV2Classifier,MAEClassifier
from src.data_processing import SpectrogramImageDataset
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Subset, ConcatDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold, StratifiedKFold
from scripts.experiments.helper import grouperf, grouper_distribution 

def print_confusion_matrix(confusion_matrix, class_names, output_dir, experiment_name, all_labels=None, all_predictions=None):
        # Ensure the logs folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    # Save the figure with a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"{output_dir}/experiment_log_{timestamp}_confusion_matrix_{experiment_name}.png"
    plt.savefig(file_name, bbox_inches="tight", dpi=300)

    print(f"\nConfusion Matrix - {experiment_name}:")
    header = "     " + "   ".join(f"{name:>4}" for name in class_names)
    print(header)
    for label, row in zip(class_names, confusion_matrix):
        row_data = "   ".join(f"{val:>4}" for val in row)
        print(f"{label:>4} {row_data}")

    # Print the path where the heatmap was saved
    print(f"Confusion matrix heatmap saved to: {file_name}")

    # Optionally print detailed label comparisons
    # if all_labels is not None and all_predictions is not None:
    #     print("\nDetailed Label Comparisons:")
    #     print(f"{'Index':<8}{'True Label':<12}{'Predicted Label':<15}")
    #     for idx, (true, pred) in enumerate(zip(all_labels, all_predictions)):
    #         true_label = class_names[true]
    #         pred_label = class_names[pred]
    #         print(f"{idx:<8}{true_label:<12}{pred_label:<15}")

    # Show the heatmap
    plt.show()
    
def resubstitution_test(model, dataset, num_epochs, lr, class_names):
    # Set up data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Move model to GPU if available
    model = model.to('cuda')
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    # Training loop
    print('Starting Resubstitution Test Training...')
    #print(dataset.get_dataset_name())
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            logits, attentions = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Average loss for the epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

    # Evaluation phase
    print('Resubstitution Evaluation...')
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            logits, attentions = model(images)
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    cm = confusion_matrix(all_labels, all_predictions)
    print(f'Resubstitution Accuracy: {accuracy:.2f}%')
    #print('Confusion Matrix:\n', cm)
    #print(dataset.get_dataset_name())
    print_confusion_matrix(cm, class_names, all_labels, all_predictions)

def one_fold_with_bias(model, dataset, num_epochs, lr, class_names):
    # Split the data with bias (random train-test split)
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=dataset.targets, random_state=42)
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False)
    
    # Move model to GPU if available
    model = model.to('cuda')
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    # Training loop
    print('Starting One-Fold (With Bias) Training...')
    #print(dataset.get_dataset_name())
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            logits, attentions = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')
    print('Training completed. Starting evaluation...')
    
    # Evaluation
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            logits, attentions = model(images)
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    cm = confusion_matrix(all_labels, all_predictions)
    print(f'One-Fold (With Bias) Accuracy: {accuracy:.2f}%')
    #print('Confusion Matrix:\n', cm)
    #print(dataset.get_dataset_name())
    print_confusion_matrix(cm, class_names, all_labels, all_predictions)

def one_fold_without_bias(model, dataset, num_epochs, lr, class_names):
    # Stratified split to reduce bias
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    X = np.arange(len(dataset))
    y = dataset.targets
    for train_idx, test_idx in sss.split(X, y):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False)
    
    # Move model to GPU if available
    model = model.to('cuda')
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    # Training loop
    print('Starting One-Fold (Without Bias) Training...')
#    print(dataset.get_dataset_name())
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            logits, attentions = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')
    print('Training completed. Starting evaluation...')
    
    # Evaluation
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            logits, attentions = model(images)
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    cm = confusion_matrix(all_labels, all_predictions)
    print(f'One-Fold (Without Bias) Accuracy: {accuracy:.2f}%')
    #print('Confusion Matrix:\n', cm) 
    #print(dataset.get_dataset_name())
    print_confusion_matrix(cm, class_names, all_labels, all_predictions)

def load_datasets(root_dir, first_datasets_name, target_datasets_name, use_domain_split, domain_name, transform):
    """
    Load datasets with proper train-validation-test splits.
    """
    train_datasets, val_datasets, test_dataset = [], [], None

    for ds_name in first_datasets_name + target_datasets_name:
        if ds_name == "UORED":
            dataset_instance = UORED(use_domain_split, domain_name)
            base_path = os.path.join(root_dir, "uored")

            if use_domain_split:
                all_domains = dataset_instance.get_all_domains()
                train_domains = [d for d in all_domains if d != domain_name]  # Exclude test domain
                test_domain = domain_name  # Only one test domain

                # Train and Validation Data
                for train_domain in train_domains:
                    train_val_path = os.path.join(base_path, train_domain)
                    if os.path.exists(train_val_path) and os.listdir(train_val_path):
                        dataset = ImageFolder(train_val_path, transform=transform)
                        train_size = int(0.8 * len(dataset))
                        val_size = len(dataset) - train_size
                        train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])
                        train_datasets.append(train_subset)
                        val_datasets.append(val_subset)
                    else:
                        print(f"[WARNING] Skipping empty domain folder: {train_val_path}")

                # Test Data
                test_path = os.path.join(base_path, test_domain)
                if os.path.exists(test_path) and os.listdir(test_path):
                    test_dataset = ImageFolder(test_path, transform=transform)
                else:
                    raise FileNotFoundError(f"Error: Test dataset folder {test_path} is missing or empty!")

            else:
                dataset_path = os.path.join(base_path, "default")
                if os.path.exists(dataset_path) and os.listdir(dataset_path):
                    dataset = ImageFolder(dataset_path, transform=transform)
                    train_size = int(0.8 * len(dataset))
                    val_size = int(0.1 * len(dataset))
                    test_size = len(dataset) - train_size - val_size
                    train_subset, val_subset, test_subset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
                    train_datasets.append(train_subset)
                    val_datasets.append(val_subset)
                    test_dataset = test_subset
                else:
                    raise FileNotFoundError(f"Error: Dataset folder {dataset_path} is missing or empty!")

        elif ds_name == "CWRU":
            dataset_path = os.path.join(root_dir, ds_name.lower())
            if os.path.exists(dataset_path) and os.listdir(dataset_path):
                dataset = ImageFolder(dataset_path, transform=transform)
                train_size = int(0.8 * len(dataset))
                val_size = int(0.1 * len(dataset))
                test_size = len(dataset) - train_size - val_size
                train_subset, val_subset, test_subset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
                train_datasets.append(train_subset)
                val_datasets.append(val_subset)
                test_dataset = test_subset
            else:
                raise FileNotFoundError(f"Error: Dataset folder {dataset_path} is missing or empty!")

    return train_datasets, val_datasets, test_dataset
    
def kfold_cross_validation(
    model,
    model_type,
    test_loader,
    num_epochs,
    lr,
    group_by="",
    class_names=[],
    n_splits=4,
    perform_kfold=True, 
    debug=False,
    datasets_name=None,
    train_domains=None,  # Multiple domains for Train/Validation
    test_domain=None  # Single domain for Test
):
    datasets_str = ", ".join(datasets_name) if datasets_name else "Unknown Dataset"
   
    batch_size = test_loader.batch_size
    dataset = test_loader.dataset

    if not perform_kfold:
        # Direct evaluation mode
        print(f"Performing direct evaluation of: {datasets_str}")
        evaluate_model(model, test_loader, class_names, debug=debug)
        return

    # K-Fold Cross-Validation mode
    y = [label for _, label in dataset]  # Extract labels
    X = np.arange(len(y))  # Indices as features

    # Determine cross-validation strategy
    if group_by:
        groups = grouperf(dataset, group_by)
        skf = StratifiedGroupKFold(n_splits=n_splits)
        split = skf.split(X, y, groups)
    else:
        if debug:
            print("Group by: None")
        skf = StratifiedKFold(n_splits=n_splits)
        split = skf.split(X, y)

    # Save initial model state
    initial_state = copy.deepcopy(model.state_dict())
    fold_metrics = []
    results_list = []

    print(f"Learning Rate: {lr}")
    #print(f"Starting K-Fold Cross-Validation... Model: {model_type.lower()}")
    print(f"Starting K-Fold Cross-Validation of: {datasets_str}")
    
    # Initialize class_sample_indices before K-Fold
    class_sample_indices = None

    for fold, (train_idx, test_idx) in enumerate(split):
        print(f"\nFold {fold + 1}/{n_splits}")

        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"Skipping Fold {fold + 1}: Empty train or test set.")
            continue
        
        # **Split train into train and validation (80-20)**
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, stratify=[y[i] for i in train_idx], random_state=42
        )

        # Prepare DataLoaders
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

        if debug:
            analyze_loader_distribution(train_loader, test_loader, class_names, fold)

        # Reset model and optimizer
        model.load_state_dict(copy.deepcopy(initial_state))
        optimizer = create_optimizer(model, lr)
        
        # Train the model with validation
        train_acc_list, val_acc_list = train_model(model, train_loader, val_loader, optimizer, num_epochs)

        attention_print = False
        if fold == 3 : attention_print = True
        # Evaluate on test set
        test_metrics, confusion_mat, all_labels, all_predictions, class_sample_indices = evaluate_model(
            model, test_loader, class_names, debug, class_sample_indices=class_sample_indices, attention_print=attention_print
        )
        fold_metrics.append(test_metrics)

        experiment_name = f"{model_type.lower()}_experiment_fold{fold + 1}"
        # Display confusion matrix
        #if attention_print:
        print_confusion_matrix(confusion_mat, class_names, output_dir="logs", 
                            experiment_name=experiment_name, all_labels=all_labels, 
                            all_predictions=all_predictions )
        results_list.append({
            "Fold": fold + 1,
            "Train Accuracy": train_acc_list[-1],  # Last epoch accuracy
            "Validation Accuracy": val_acc_list[-1],  # Last epoch validation accuracy
            "Test Accuracy": test_metrics["accuracy"],
        })


    # Summarize results across folds
    mean_metrics = summarize_kfold_results(fold_metrics)

    # Compute Mean Accuracies for CSV
    mean_train_acc = np.mean([r['Train Accuracy'] for r in results_list])
    mean_val_acc = np.mean([r['Validation Accuracy'] for r in results_list])
    mean_test_acc = mean_metrics["accuracy"]

    # Print final summary
    print("\n **Final K-Fold Cross-Validation Summary**")
    print(f"   **Mean Train Accuracy:** {mean_train_acc:.2f}%")
    print(f"   **Mean Validation Accuracy:** {mean_val_acc:.2f}%")
    print(f"   **Mean Test Accuracy:** {mean_test_acc:.2f}%")
    
    # Return values to be saved in CSV
    return {
        **mean_metrics,
        "train_accuracy": mean_train_acc,
        "validation_accuracy": mean_val_acc,
        "test_accuracy": mean_test_acc      
    }    
    #evaluate_full_model(model,test_loader) - don't use this as it's getting the last kfold weights 

def kfold_validation(
    model,
    model_type,
    train_val_loader,
    test_loader,
    num_epochs,
    lr,
    class_names=[],
    n_splits=4,
    debug=False,
    patience=5  # 🔧 New: Early stopping patience
):

    print("Starting K-Fold Validation (with separate test set)...")
    batch_size = train_val_loader.batch_size
    dataset = train_val_loader.dataset

    y = [label for _, label in dataset]
    X = np.arange(len(y))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = skf.split(X, y)

    initial_state = copy.deepcopy(model.state_dict())
    fold_metrics = []
    results_list = []
    sample_idx = None

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\nFold {fold + 1}/{n_splits}")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        if debug:
            analyze_loader_distribution(train_loader, val_loader, class_names, fold)

        model.load_state_dict(copy.deepcopy(initial_state))
        optimizer = create_optimizer(model, lr)

        train_acc_list, val_acc_list, best_train_acc, best_val_acc = train_model(
            model, train_loader, val_loader, optimizer, num_epochs,
            patience=patience  # Early stopping enabled
        )

        #print(f"Fold {fold + 1} Results → Train Acc: {train_acc_list[-1]:.2f}%, Val Acc: {val_acc_list[-1]:.2f}%")
        print(f"Fold {fold + 1} Results → Best Train Acc: {best_train_acc:.2f}%, Best Val Acc: {best_val_acc:.2f}%")

        sample_idx = val_idx[0]
        
        results_list.append({
            "Fold": fold + 1,
            "Train Accuracy": best_train_acc,
            "Validation Accuracy": best_val_acc,
        }), 0

    mean_train_acc = np.mean([r['Train Accuracy'] for r in results_list])
    mean_val_acc = np.mean([r['Validation Accuracy'] for r in results_list])

    print("\n **Final Cross-Validation Summary (Train/Val)**")
    print(f"   **Mean Train Accuracy:** {mean_train_acc:.2f}%")
    print(f"   **Mean Validation Accuracy:** {mean_val_acc:.2f}%")

    print("\n>> Final Evaluation on Independent Test Set")
    attention_print = False
    
    if sample_idx is not None:
        attention_print = True
    
    final_metrics, cm, all_labels, all_predictions, _ = evaluate_model(
        model, test_loader, class_names, debug=debug, sample_idx=sample_idx, attention_print=attention_print 
    )
    print_confusion_matrix(cm, class_names, output_dir="logs",
                           experiment_name=f"{model_type.lower()}_final_test",
                           all_labels=all_labels,
                           all_predictions=all_predictions)

    return {
        **final_metrics,
        "train_accuracy": mean_train_acc,
        "validation_accuracy": mean_val_acc,
        "test_accuracy": final_metrics["accuracy"]
    }

def analyze_loader_distribution(train_loader, test_loader, class_names, fold):
    """Analyzes the class distribution in the DataLoaders."""
    for loader_name, loader in zip(["Train", "Test"], [train_loader, test_loader]):
        labels = [label for _, label in loader.dataset]
        distribution = {class_names[i]: labels.count(i) for i in range(len(class_names))}
        print(f"Fold {fold + 1} - {loader_name} Distribution: {distribution}")

def create_optimizer(model, lr):
    """Creates an optimizer based on the model type."""
    if isinstance(model, DeiTClassifier):
        params = model.deit.parameters()
    elif isinstance(model, ViTClassifier):
        params = model.vit.parameters()
    elif isinstance(model, DINOv2WithRegistersClassifier):
        params = model.dinov2.parameters()
    elif isinstance(model, SwinV2Classifier):
        params = model.swinv2.parameters()
    elif isinstance(model, MAEClassifier):
        params = model.mae.parameters()
    elif isinstance(model, CNN2D):
        params = model.parameters()  
    elif isinstance(model, ResNet18):
        params = model.parameters()      
    else:
        raise ValueError("Unsupported model type.")

    return AdamW(params, lr=lr, weight_decay=1e-4)

def train_model(model, train_loader, val_loader, optimizer, num_epochs, patience=5, lr_decay_step=5, lr_decay_factor=0.5):
    """
    Train the model while logging training and validation accuracy.
    """
    criterion = nn.CrossEntropyLoss()
    train_acc_list, val_acc_list = [], []

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        correct_train, total_train, running_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):  #Extract logits if a tuple is returned
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            running_loss += loss.item()

        train_accuracy = 100 * correct_train / total_train
        train_acc_list.append(train_accuracy)

        # Validation
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to("cuda"), labels.to("cuda")
                outputs = model(images)
                if isinstance(outputs, tuple):  #Extract logits if a tuple is returned
                    outputs = outputs[0]
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        val_acc_list.append(val_accuracy)

        # Early Stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_train_acc = train_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"--- Early stopping triggered at epoch {epoch+1} (Best Val Acc: {best_val_acc:.2f}%)")
                break

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

    return train_acc_list, val_acc_list, best_train_acc, best_val_acc

def evaluate_model(model, test_loader, class_names, debug=False, sample_idx=None, attention_print=False):
    """Evaluates the model and returns metrics, confusion matrix, labels, and predictions."""
    model.eval()
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to("cuda"), labels.to("cuda")
            logits, attentions = model(images)
            _, predictions = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    precision = precision_score(all_labels, all_predictions, average="weighted") * 100
    recall = recall_score(all_labels, all_predictions, average="weighted") * 100
    f1 = f1_score(all_labels, all_predictions, average="weighted") * 100
    cm = confusion_matrix(all_labels, all_predictions)

    if debug:
        print("Final Test Evaluation Report:")
        print(classification_report(all_labels, all_predictions, target_names=class_names,zero_division=1))

    if hasattr(model, 'vit') or hasattr(model, 'deit') or hasattr(model, 'dinov2') or hasattr(model, 'swinv2') or hasattr(model, 'mae'):
        if attention_print: 
            # Visualize attention for representative samples of each class
            class_sample_indices = pick_representative_indices_by_class(test_loader, num_classes=len(class_names))

            for class_id, sample_idx in class_sample_indices.items():
                print(f"Class {class_names[class_id]} >> Index {sample_idx}")
                visualize_attention(
                    dataset=test_loader.dataset,
                    model=model,
                    idx=sample_idx,
                    attentions=None,
                    head=0,
                    layer=-1
                )
    else:
         print("\nSkipping attention visualization as the model does not support it.")

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return metrics, cm, all_labels, all_predictions, None

def summarize_kfold_results(fold_metrics):
    """Summarizes the results across folds."""
    if not fold_metrics:
        print("No valid folds to summarize.")
        return

    mean_metrics = {key: np.mean([fold[key] for fold in fold_metrics]) for key in fold_metrics[0]}
    print("\nK-Fold Cross-Validation Results:")
    for key, value in mean_metrics.items():
        print(f"  - Mean {key.capitalize()}: {value:.2f}%")
    
    return mean_metrics

def pick_representative_indices_by_class(test_loader, num_classes=4):
    """
    Pick one index per class from test_loader.dataset.
    Returns: dict {class_id: sample_index}
    """
    dataset = test_loader.dataset
    representative_indices = {}

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)

        if label not in representative_indices:
            representative_indices[label] = idx

        if len(representative_indices) == num_classes:
            break

    return representative_indices


 
def evaluate_full_model(model, test_loader):
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits, _ = model(images.to('cuda'))
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    report = classification_report(all_labels, all_predictions, zero_division=1)
    print("\nFinal Test Evaluation Report (eval using Test-Loader after last full KFold experiment):")
    print(report)

def create_balanced_dataloader(dataset, batch_size):
    # Compute class counts and weights
    class_counts = np.bincount([label for _, label in dataset])
    class_weights = 1.0 / class_counts

    # Assign sample weights
    sample_weights = [class_weights[label] for _, label in dataset]

    # Define the sampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    print("Balanced DataLoader created.")
    return dataloader
        
def visualize_attention(dataset, model, idx, attentions, head=0, layer=-1):
    """
    Visualize attention maps for a spectrogram from the dataset.
    Args:
        dataset (Dataset): The spectrogram dataset (SpectrogramImageDataset).
        model (ViTClassifier or DeiT): The trained Vision Transformer model.
        idx (int): Index of the spectrogram in the dataset.
        attentions: Attention outputs from the model.
        head (int): Attention head to visualize.
        layer (int): Layer to extract attention from (-1 for the last layer).
    """
    # Retrieve the spectrogram and label
    spectrogram, label = dataset[idx]
    
    # Define class mapping (adjust if necessary)
    class_mapping = {0: 'B', 1: 'N', 2: 'O', 3: 'I'}
    
    # Convert the label index to class name
    class_name = class_mapping.get(label, "Unknown")
    
    # Convert the spectrogram to a tensor and preprocess it
    image_tensor = spectrogram.unsqueeze(0).to(model.device)  # Already transformed

    # Forward pass through the model to get attentions
    logits, attentions = model(image_tensor)
    
    # Handle missing attentions
    if attentions is None or layer >= len(attentions):
        print(f"Attention output is None or invalid for layer {layer}.")
        return

    # Extract the attention map for the specified layer and head
    attention_map = attentions[layer][0, head, :, :]  # Shape: [seq_len, seq_len]

    # Average over rows (token queries)
    aggregated_attention = attention_map.mean(dim=0).detach().cpu().numpy()

    # Reshape the attention map to match spectrogram dimensions
    height, width = image_tensor.shape[-2], image_tensor.shape[-1]
    attention_resized = cv2.resize(aggregated_attention, (width, height), interpolation=cv2.INTER_LINEAR)

    # Normalize the attention map for visualization
    attention_resized = (attention_resized - np.percentile(attention_resized, 5)) / (
        np.percentile(attention_resized, 95) - np.percentile(attention_resized, 5)
    )
    # attention_resized = np.clip(attention_resized, 0, 1)
    # attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())

    # Convert spectrogram to numpy for visualization
    spectrogram_np = np.asarray(spectrogram).astype(float)
    if spectrogram_np.shape[0] == 3:  # Check if it's an RGB image
        spectrogram_np = np.transpose(spectrogram_np, (1, 2, 0))  # Convert to HWC format

    vmin = np.percentile(attention_resized, 5)
    vmax = np.percentile(attention_resized, 95)
    
    # Normalize spectrogram data to [0, 1]
    if spectrogram_np.max() > 1.0:
        spectrogram_np = (spectrogram_np - spectrogram_np.min()) / (spectrogram_np.max() - spectrogram_np.min())
    
    # print("Spectrogram min and max:", spectrogram_np.min(), spectrogram_np.max())
    # print("Attention map min and max:", attention_resized.min(), attention_resized.max())
    
    # Plot spectrogram and attention map side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Attention Visualization for Sample {idx} - Class: {class_name} - Label: {label}", fontsize=16)

    # Spectrogram
    spectrogram_im = axs[0].imshow(spectrogram_np, cmap="jet", aspect='auto', origin='lower') # Force auto aspect ratio
    #axs[0].imshow(attention_resized, cmap="plasma", alpha=0.5, aspect="auto") #Overlay attention on spectrogram
    axs[0].set_title("Original Spectrogram", fontsize=14)
    axs[0].set_ylabel("Frequency (Hz)", fontsize=12)
    axs[0].set_xlabel("Time (s)", fontsize=12)
    axs[0].axis("on")  # Ensure axes are shown

    # Add color bar for the spectrogram
    cbar_spec = fig.colorbar(spectrogram_im, ax=axs[0], fraction=0.046, pad=0.04)
    cbar_spec.set_label("Spectrogram Intensity", fontsize=12, rotation=270, labelpad=20)

    # Attention Map
    im = axs[1].imshow(attention_resized, cmap="plasma", aspect='auto', vmin=0, vmax=1)
    threshold = 0.7  # High attention threshold
    binary_mask = attention_resized > threshold
    axs[1].contour(binary_mask, colors=['black', 'gray'], linewidths=0.3, linestyles='dotted') #.contour(attention_resized, levels=5, colors="white", linewidths=0.5)  
    #axs[1].contour(binary_mask, levels=5, colors="white", linewidths=0.5)
    axs[1].set_title("Attention Map", fontsize=14)
    axs[1].set_ylabel("Attention Head Tokens", fontsize=12)
    axs[1].set_xlabel("Sequence Tokens", fontsize=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", fontsize=12, rotation=270, labelpad=20)

    # Layout adjustment
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure to the logs folder
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)  # Ensure logs folder exists
    file_name = f"{log_folder}/experiment_log_{timestamp}_{class_name}_{idx}.png"
    fig.savefig(file_name, bbox_inches="tight",dpi=300)  # Save the figure using fig object

    plt.show()
    plt.close(fig)  # Explicitly close the figure to free memory
    print(f"Saved attention visualization to {file_name}")