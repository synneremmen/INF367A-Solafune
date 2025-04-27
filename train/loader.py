import sys
from OBA.object_based_augmentation import create_OBA_tensor_dataset
from utils.preprocessing import get_processed_data
sys.path.append('../../..')
from torch.utils.data import DataLoader, random_split
from torch import Generator

def get_loader(dataset, batch_size, train_split=0.7, val_split=0.2, test_split=0.1, seed_value=42):
    assert int(train_split * 100) + int(val_split * 100) + int(test_split * 100) == 100, f"Splits must sum to 1, but got {train_split + val_split + test_split}"
    
    seed = Generator().manual_seed(seed_value) 
    
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train, val, test = random_split(dataset, [train_size, val_size, test_size], generator=seed)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_dataset(dataset_type, subset=False):
    print("\n\nLoading data...")
    use_OBA = False
    use_SR = False

    if dataset_type == "normal":
        dataset = get_processed_data(subset=subset)
    elif dataset_type == "OBA":
        use_OBA = True
        use_SR = False
    elif dataset_type == "SR":
        use_OBA = False
        use_SR = True
    elif dataset_type == "SR_OBA":
        use_OBA = True
        use_SR = True
    else:
        raise ValueError("Invalid dataset type. Choose from 'normal', 'OBA', 'SR', or 'SR_OBA'.")
    
    if use_OBA:
        print(f"Using OBA{' and SR' if use_SR else ''} dataset")
        dataset = create_OBA_tensor_dataset(
            prob_of_OBA=0.5, # how much OBA data to generate
            subset=True,
            augm=True,
            object_augm=True,
            extra_background_prob=0, # not in use
            background_augm_prob=0.6,
            shadows=False, # not to be used
            extra_objects=3,
            object_augm_prob=0.6,
            augm_prob=0.8,
            geometric_augm_prob=0.6,
            color_augm_prob=0.6,
            batch_size=10,
            min_area=1000, # how much area to be considered as an object
            use_SR=use_SR,
        )

    if use_SR:
        print("Using SR dataset")
        dataset = get_processed_data(use_SR=True, subset=False)

    return dataset