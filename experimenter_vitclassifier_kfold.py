import torch
import os
import copy
import sys
import logging
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.models import CNN2D, ViTClassifier, ResNet18, DeiTClassifier, DINOv2WithRegistersClassifier, SwinV2Classifier, MAEClassifier #, DeepSeekVL2Classifier 
from src.models.vitclassifier import train_and_save, load_trained_model
from scripts.evaluate_model_vitclassifier import kfold_validation, resubstitution_test, one_fold_with_bias, one_fold_without_bias, evaluate_full_model
from datasets.uored import UORED
from datasets.cwru import CWRU
from imblearn.over_sampling import SMOTE
from torch.utils.data import TensorDataset
from utils.logginout import LoggerWriter
from utils.print_info import print_info


def apply_smote_to_imagefolder(imagefolder_dataset):
    """
    Applies SMOTE to an ImageFolder dataset by flattening image tensors.
    Returns a new TensorDataset with balanced samples.
    """
    images = []
    labels = []

    for img, label in imagefolder_dataset:
        images.append(img.view(-1).numpy())  # Flatten image
        labels.append(label)

    smote = SMOTE()
    X_res, y_res = smote.fit_resample(np.array(images), np.array(labels))

    # Reshape back to image tensors (assumes original size 3x224x224)
    X_res_tensors = torch.tensor(X_res).float().view(-1, 3, 224, 224)
    y_res_tensors = torch.tensor(y_res).long()

    print(f"[INFO] Applied SMOTE: Original={len(images)}, After={len(X_res)}")
    return TensorDataset(X_res_tensors, y_res_tensors)

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

def enforce_consistent_mapping(datasets: list[ImageFolder], desired_class_to_idx: dict):
    """
    Ensures that all datasets have the same class-to-index mapping.
    Args:
        datasets (list): List of ImageFolder datasets.
        desired_class_to_idx (dict): The desired class-to-index mapping.
    """

    for dataset in datasets:
        dataset_class_to_idx = dataset.class_to_idx
        
        # Identify unexpected classes
        unexpected_classes = set(dataset_class_to_idx.keys()) - set(desired_class_to_idx.keys())
        if unexpected_classes:
            print(f"[WARNING] Ignoring unexpected classes in dataset: {unexpected_classes}")

        # Create new mapping for valid classes only
        valid_samples = []
        for path, label in dataset.samples:
            class_name = dataset.classes[label]  # Get class name from current label

            # Skip samples that belong to unexpected classes
            if class_name not in desired_class_to_idx:
                continue  

            new_label = desired_class_to_idx[class_name]  # Map to desired label
            valid_samples.append((path, new_label))

        # Update dataset with only valid samples
        dataset.samples = valid_samples
        dataset.class_to_idx = desired_class_to_idx
        dataset.classes = sorted(desired_class_to_idx, key=desired_class_to_idx.get)
        #dataset.classes = list(desired_class_to_idx.keys())
        print(f"[INFO] Dataset updated: {len(valid_samples)} samples retained for consistent classes.")
        print(f"[INFO] Labels in dataset after mapping: {Counter([label for _, label in dataset.samples])}")


    print("[INFO] Mappings enforced successfully.")
    
def experimenter_classifier_v2(
    model_type="DeiT",
    pretrain_model=False,
    base_model=True,
    num_classes=4,
    num_epochs=20,
    lr=0.00005,
    num_epochs_kf=10,
    lr_kf=0.00005,
    batch_size=32,
    root_dir="data/spectrograms",
    dataset_name="UORED",
    perform_kfold=True,
    mode="supervised",
    use_domain_split=False,
    train_domains=None,
    test_domain=None,
    num_segments=20,
    use_SMOTE=False
):
    print(f"Experiment Parameters: {locals()}")

    # === Model setup ===
    model_classes = {
        "ViT": ViTClassifier,
        "DeiT": DeiTClassifier,
        "DINOv2": DINOv2WithRegistersClassifier,
        "SwinV2": SwinV2Classifier,
        "MAE": MAEClassifier,
        "CNN2D": CNN2D,
        "ResNet18": ResNet18
    }
    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_class = model_classes[model_type]
    model = model_class(num_classes=num_classes).to("cuda")

    # === Class mapping and transforms ===
    class_to_idx = {"B": 0, "I": 1, "N": 2, "O": 3}
    class_names = list(class_to_idx.keys())

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # === Path handling ===
    dataset_classes = {"UORED": UORED, "CWRU": CWRU}
    if dataset_name not in dataset_classes:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_obj = dataset_classes[dataset_name](
        use_domain_split=use_domain_split,
        train_domains=train_domains,
        test_domain=test_domain
    )

    train_subdir = dataset_obj.get_domain_folder(is_test=False)
    test_subdir = dataset_obj.get_domain_folder(is_test=True)
    train_path = os.path.join(root_dir, dataset_name.lower(), train_subdir)
    test_path = os.path.join(root_dir, dataset_name.lower(), test_subdir)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing training path: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test path: {test_path}")

    full_train_dataset = ImageFolder(train_path, transform=transform)
    test_dataset = ImageFolder(test_path, transform=transform)

    enforce_consistent_mapping([full_train_dataset, test_dataset], class_to_idx)
    
    print("[DEBUG] Final label distribution in test_dataset:")
    print(Counter([label for _, label in test_dataset.samples]))

    train_val_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"[DEBUG] Data split - Total Train+Val: {len(full_train_dataset)}, Test: {len(test_dataset)}")

    if not base_model:
        raise NotImplementedError("Pre-trained model strategy is not implemented in this version.")

    print(f"Using base model {model_type}Classifier with no pre-training.")

    # === Run K-Fold with separate test ===
    metrics = kfold_validation(
        model=model,
        model_type=model_type,
        train_val_loader=train_val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs_kf,
        lr=lr_kf,
        class_names=class_names,
        n_splits=4,
        debug=True,
        patience=6
    )

    return metrics


def run_experimenter():     
    experimenter_classifier_v2(
        model_type="ViT",  
        pretrain_model=False,
        base_model=True,
        num_classes=4,
        num_epochs=10,
        lr=0.00005,
        batch_size=32,
        root_dir="data/spectrograms",
        dataset_name="UORED",
        perform_kfold=True,
        mode="supervised",  # "pretrain", "supervised", or "both"
        use_domain_split= False,
        domain_name= None
    )

if __name__ == "__main__":
    #sys.stdout = LoggerWriter(logging.info, "kfold-vitclassifier")
    run_experimenter()
    
    
