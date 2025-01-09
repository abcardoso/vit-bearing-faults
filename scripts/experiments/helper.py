import os
from src.data_processing import DatasetManager
from torch.utils.data import DataLoader, ConcatDataset

annot = DatasetManager()

# Map feature values to class labels (N, B, O, I)
class_label_mapping = {
    "N": "Normal",
    "B": "Ball Fault",
    "O": "Outer Race Fault",
    "I": "Inner Race Fault"
}

# Compute and print class distribution for train and test splits
def grouper_distribution(dataset, feature_mitigation, indices, class_names):
    """
    Computes class distribution using the grouper logic.

    Args:
        dataset: The dataset object (supports ConcatDataset and standard datasets).
        feature_mitigation (str): Feature to group by (optional).
        indices (list): List of indices in the dataset to compute distribution.
        class_names (list): List of class names corresponding to label indices.

    Returns:
        dict: A dictionary with class names as keys and counts as values.
    """
    if not feature_mitigation:
        print('Group by: none')
        feature_mitigation = "label"  # Default to label if no feature is specified

    # Extract samples from ConcatDataset or standard dataset
    if isinstance(dataset, ConcatDataset):
        samples = []
        for sub_dataset in dataset.datasets:
            samples.extend(sub_dataset.samples)
    else:
        samples = dataset.samples

    # Initialize class distribution dictionary
    class_distribution = {class_name: 0 for class_name in class_names}

    # Traverse the specified indices to calculate distribution
    for idx in indices:
        path = samples[idx][0]
        basename = os.path.basename(path).split('#')[0]
        bearing_info = annot.filter_data({"filename": basename})[0]
        feature_value = bearing_info.get(feature_mitigation, "default")
        class_label = bearing_info.get("label", "default")

        # Increment class count if the feature value is in class names
        if feature_value in class_names:
            class_distribution[feature_value] += 1

    # Print the computed class distribution
    print(f'Computed Class Distribution by {feature_mitigation}:', class_distribution)
    return class_distribution

def get_class_counter(dataset, feature_label="label", verbose=True):
    """
    Counts the number of samples per class in the dataset.
    Supports both standard datasets and ConcatDataset.
    """
    class_counter = {}

    # Handle both standard datasets and ConcatDataset
    samples = (
        [sample for sub_dataset in dataset.datasets for sample in sub_dataset.samples]
        if isinstance(dataset, ConcatDataset)
        else dataset.samples
    )

    # Iterate through samples and count class occurrences
    for sample in samples:
        basename = os.path.basename(sample[0]).split('#')[0]
        try:
            bearing_info = annot.filter_data({"filename": basename})[0]
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            continue

        class_label = bearing_info.get(feature_label, "default")
        class_counter[class_label] = class_counter.get(class_label, 0) + 1

    # Print the total samples per class if verbose
    if verbose:
        counter_str = " | ".join([f"{class_key}: {count}" for class_key, count in class_counter.items()])
        print(f"Total samples per class ({feature_label}) | {counter_str}")

    return class_counter


def get_counter(dataset, feature_mitigation, default_value="default", verbose=True):
    """
    Calculates occurrences of a specific feature in the dataset.
    Handles both simple datasets and ConcatDataset.
    """
    counter = {}

    # Handle both standard datasets and ConcatDataset
    samples = (
        [sample for sub_dataset in dataset.datasets for sample in sub_dataset.samples]
        if isinstance(dataset, ConcatDataset)
        else dataset.samples
    )

    # Iterate through samples and count feature occurrences
    for sample in samples:
        basename = os.path.basename(sample[0]).split('#')[0]
        try:
            bearing_info = annot.filter_data({"filename": basename})[0]
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            continue

        feature_value = bearing_info.get(feature_mitigation, default_value)
        counter[feature_value] = counter.get(feature_value, 0) + 1

    # Print the feature-wise counter if verbose
    if verbose:
        counter_str = " | ".join([f"{feature}: {count}" for feature, count in counter.items()])
        print(f"Feature-wise counter for '{feature_mitigation}' | {counter_str}")

    return counter

def grouper(dataset, feature_mitigation):
    """
    Groups the dataset based on a specific feature for mitigation purposes.
    Supports both standard datasets and ConcatDataset.
    """
    if not feature_mitigation:
        print('Group by: none')
        # If `feature_mitigation` is empty, return a default group for all items
        return [0] * len(dataset)

    # Extract samples from ConcatDataset or standard dataset
    if isinstance(dataset, ConcatDataset):
        samples = []
        for sub_dataset in dataset.datasets:
            samples.extend(sub_dataset.samples)
    else:
        samples = dataset.samples

    # Initialize group metadata
    groups = []
    counter = get_counter(dataset, feature_mitigation)
    class_counter = get_class_counter(dataset, "label")
    num_groups = len(counter)

    # Initialize group class distribution tracker
    group_class_distribution = {g: {label: 0 for label in class_counter} for g in range(num_groups)}

    # Assign samples to groups
    for i in range(len(samples)):
        path = samples[i][0]
        basename = os.path.basename(path).split('#')[0]
        bearing_info = annot.filter_data({"filename": basename})[0]
        feature_value = bearing_info.get(feature_mitigation, "default")
        class_label = bearing_info.get("label", "default")

        # Assign sample to an appropriate group
        assigned = False
        for group, distribution in group_class_distribution.items():
            if distribution[class_label] < class_counter[class_label] // num_groups:
                group_class_distribution[group][class_label] += 1
                groups.append(group)
                assigned = True
                break

        if not assigned:
            # Assign to a random group if balancing fails
            random_group = len(groups) % num_groups
            group_class_distribution[random_group][class_label] += 1
            groups.append(random_group)

    # Print detailed group information
    print('Group by:', feature_mitigation)
    print('Groups:', set(groups))
    print('Counter:', counter)
    print('ClassCounter:', class_counter)
    print('Group Class Distribution:')
    for group, distribution in group_class_distribution.items():
        print(f"  Group {group}: {distribution}")

    return groups

def grouperf(dataset, feature_mitigation): 
    """
    Groups the dataset based on a specific feature for mitigation purposes.
    Supports both standard datasets and ConcatDataset.
    """
    if not feature_mitigation:
        print("Group by: none")
        return [0] * len(dataset)

    # Extract samples
    samples = []
    if isinstance(dataset, ConcatDataset):
        for sub_dataset in dataset.datasets:
            samples.extend(sub_dataset.samples)
    else:
        samples = dataset.samples

    # Metadata initialization
    groups = []
    counter = get_counter(dataset, feature_mitigation)
    class_counter = get_class_counter(dataset, "label")
    num_groups = len(counter)

    # Group class distribution initialization
    group_class_distribution = {g: {label: 0 for label in class_counter} for g in range(num_groups)}

    # Assign samples to groups
    for i, sample in enumerate(samples):
        path = sample[0]
        basename = os.path.basename(path).split('#')[0]
        bearing_info = annot.filter_data({"filename": basename})[0]
        class_label = bearing_info.get("label", "default")

        # Assign sample to a group with the least filled slot for the class
        least_filled_group = min(group_class_distribution, key=lambda g: group_class_distribution[g][class_label])
        group_class_distribution[least_filled_group][class_label] += 1
        groups.append(least_filled_group)

    # Print detailed group information
    print(f"Group by: {feature_mitigation} | Groups: {set(groups)} | Counter: {counter} | ClassCounter: {class_counter}")
    for group, distribution in group_class_distribution.items():
        dist_str = " | ".join([f"{label}: {count}" for label, count in distribution.items()])
        print(f"  Group {group} | {dist_str}")

    return groups
