import torch
import os
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.models import CNN2D, ViTClassifier, ResNet18, DeiTClassifier, DINOv2WithRegistersClassifier, SwinV2Classifier, MAEClassifier #, DeepSeekVL2Classifier 
from src.models.vitclassifier import train_and_save, load_trained_model
from scripts.evaluate_model_vitclassifier import kfold_cross_validation, resubstitution_test, one_fold_with_bias, one_fold_without_bias, evaluate_full_model
from datasets.uored import UORED
from datasets.cwru import CWRU

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
        dataset.classes = list(desired_class_to_idx.keys())

    print("[INFO] Mappings enforced successfully.")


def experimenter_classifier(
    model_type="DeiT",  # Options: "ViT", "DeiT", "DINOv2", "SwinV2", "DeepSeekVL2", "MAE", "CNN2D", "ResNet18"
    pretrain_model=False,
    base_model=True,
    num_classes=4,
    num_epochs=20,
    lr=0.00005,
    num_epochs_kf=10,
    lr_kf=0.00005,
    batch_size=32,
    root_dir="data/spectrograms",
    first_datasets_name=["CWRU"],
    target_datasets_name=["UORED"],
    perform_kfold=True,
    mode="supervised",  # "pretrain", "supervised", or "both"
    use_domain_split=False,
    domain_name=None
):
    print(f"Experiment Parameters: {locals()}")
    
    # Map model_type to corresponding class
    model_classes = {
        "ViT": ViTClassifier,
        "DeiT": DeiTClassifier,
        "DINOv2": DINOv2WithRegistersClassifier,
        "SwinV2": SwinV2Classifier,
        "MAE": MAEClassifier,
        "CNN2D": CNN2D,
        "ResNet18": ResNet18
        #,"DeepSeekVL2": DeepSeekVL2Classifier
    }

    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from {list(model_classes.keys())}.")

    # Initialize the model
    model_class = model_classes[model_type]
    model = model_class(num_classes=num_classes).to("cuda")

    # Class-to-index mapping
    class_to_idx = {"B": 0, "I": 1, "N": 2, "O": 3}

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    datasets = []
    for ds_name in first_datasets_name + target_datasets_name:
        if ds_name == "UORED":
            dataset_instance = UORED(use_domain_split, domain_name)
        elif ds_name == "CWRU":
            dataset_instance = CWRU()
        else:
            raise ValueError(f"Dataset {ds_name} not recognized.")

        # Ensure correct path for UORED with domain split
        if ds_name == "UORED":
            dataset_path = os.path.join(root_dir, "uored")  # Remove domain_name from here
        else:
            dataset_path = os.path.join(root_dir, ds_name.lower())

        # Retrieve domain folder (returns "" when use_domain_split=False)
        domain_folder = dataset_instance.get_domain_folder("")

        # Append domain folder only when it is not empty
        if domain_folder:
            dataset_path = os.path.join(dataset_path, domain_folder)

        # Debugging check: Print the final dataset path to verify correctness
        print(f"[DEBUG] Loading dataset from: {dataset_path}")

        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            raise FileNotFoundError(
                f"Error: The dataset folder {dataset_path} is missing or empty!\n"
                "Ensure spectrograms are generated correctly."
            )

        # Ensure ImageFolder() is loading from a directory that contains class folders
        if not any(os.scandir(dataset_path)):
            raise FileNotFoundError(
                f"Error: The dataset folder {dataset_path} exists but is empty.\n"
                "Make sure `generate_spectrogram()` ran correctly before executing the experiment."
            )

        # Check if dataset_path needs class subfolders (Only for CWRU)
        if ds_name == "CWRU":
            dataset_path = os.path.join(root_dir, ds_name.lower())  # Ensure class directories are included

        # Debugging print statement
        print(f"[DEBUG] Final dataset path for ImageFolder: {dataset_path}") 

        dataset = ImageFolder(dataset_path, transform=transform)
        datasets.append(dataset)
        
    # Load train and test datasets
    first_datasets = [ImageFolder(os.path.join(root_dir, ds.lower()), transform) for ds in first_datasets_name]
    target_datasets = [ImageFolder(os.path.join(root_dir, ds.lower()), transform) for ds in target_datasets_name]

    # Enforce consistent mapping
    enforce_consistent_mapping(first_datasets, class_to_idx)
    enforce_consistent_mapping(target_datasets, class_to_idx)

    # Combine datasets
    first_concated_dataset = ConcatDataset(first_datasets)
    target_concated_dataset = ConcatDataset(target_datasets)

    # Split train data for k-fold
    train_idx, eval_idx = train_test_split(
        range(len(first_concated_dataset)),
        test_size=0.2,
        stratify=[y for _, y in first_concated_dataset]
    )

    # Create DataLoaders
    first_train_loader = DataLoader(
        torch.utils.data.Subset(first_concated_dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    first_eval_loader = DataLoader(
        torch.utils.data.Subset(first_concated_dataset, eval_idx),
        batch_size=batch_size, shuffle=False, num_workers=4
    )
    target_loader = DataLoader(target_concated_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    first_full_loader = DataLoader(first_concated_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Compute and print class distributions
    for dataset_name, dataset_loader in zip(first_datasets_name, [first_train_loader]):
        print(f"\n>> Calculating distribution for first dataset (if used) ({dataset_name})...")
        compute_and_print_distribution(dataset_loader, class_to_idx, dataset_name)

    for dataset_name, dataset_loader in zip(target_datasets_name, [target_loader]):
        print(f"\n>> Calculating distribution for target dataset (TARGET dataset) ({dataset_name})...")
        compute_and_print_distribution(dataset_loader, class_to_idx, dataset_name)

    # Define model save path
    saved_model_path = f"saved_models/{model_type.lower()}_classifier.pth"
    print_info("Experiment", [f"Transfer Learning with {model_type}Classifier"])
    if not base_model:
        print(f"Saved model path: {saved_model_path}")

    # Pre-train or load model
    if not base_model:
        if pretrain_model:
            print("First approach - training the model.")
            teacher_model = None
            # Check if the pretrained checkpoint exists
            pretrained_checkpoint = saved_model_path.replace(".pth", "_pretrained.pth")
            if mode == "pretrain":
                teacher_model = model_class(num_classes=num_classes).to("cuda")

            # Initialize or load the model
            if not os.path.exists(pretrained_checkpoint):
                print(f"No pretrained checkpoint found. Initializing new model for pretraining: {model_type}Classifier")
                model = model_class(num_classes=num_classes).to("cuda")
            else:
                print(f"Loading pretrained checkpoint from {pretrained_checkpoint}")
                model = load_trained_model(model_class, pretrained_checkpoint, num_classes=num_classes, pretrained=True)

            train_and_save(model, 
                           first_full_loader, #first_train_loader,  
                           first_eval_loader, #not used
                           num_epochs, lr, saved_model_path,mode=mode, 
                           pretrain_epochs=num_epochs,teacher_model=teacher_model, 
                           datasets_name = first_datasets_name)
        else:
            print("Loading pre-trained model.")
            model = load_trained_model(model_class, saved_model_path, 
                                       num_classes=num_classes, pretrained=pretrain_model).to("cuda")
    else:
        print(f"Using base model {model_type}Classifier with no pre-training.")

    # Run k-fold cross-validation
    group_by = ""  # Modify this if needed
    #print("Starting k-fold cross-validation...")
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
    
    metrics = kfold_cross_validation(
        model,
        model_type,  
        first_eval_loader if set(target_datasets_name) == set(first_datasets_name) else target_loader,
        num_epochs = num_epochs_kf,
        lr = lr_kf, 
        group_by="", 
        class_names = list(class_to_idx.keys()), 
        n_splits=4,
        perform_kfold=perform_kfold,  # New parameter to toggle K-Fold Cross-Validation
        debug=True,
        datasets_name=target_datasets_name #first_datasets_name
    )
    return metrics
    

def run_experimenter():     
    experimenter_classifier(
        model_type="ViT",  
        pretrain_model=False,
        base_model=True,
        num_classes=4,
        num_epochs=10,
        lr=0.00005,
        batch_size=32,
        root_dir="data/spectrograms",
        first_datasets_name=["CWRU"],
        target_datasets_name=["UORED"],
        perform_kfold=True,
        mode="supervised",  # "pretrain", "supervised", or "both"
        use_domain_split= False,
        domain_name= None

    
    )

if __name__ == "__main__":
    #sys.stdout = LoggerWriter(logging.info, "kfold-vitclassifier")
    run_experimenter()
    
    
