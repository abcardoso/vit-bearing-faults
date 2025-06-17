import os
from datasets import CWRU, UORED, Paderborn, Hust

def download_rawfile(dataset_name='all'):
    dataset_map = {
        "CWRU": CWRU,
        "UORED": UORED,
        "Paderborn": Paderborn,
        "Hust": Hust
    }

    if dataset_name == 'all':
        for dataset_class in dataset_map.values():
            dataset = dataset_class()
            dataset.download()
    elif dataset_name in dataset_map:
        dataset = dataset_map[dataset_name]()  # Instantiate class
        dataset.download()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")   