import sys
sys.path.append('../../..')
from torch.utils.data import DataLoader, random_split
from torch import Generator

def get_loader(dataset, batch_size, train_split=0.7, val_split=0.2, test_split=0.1, seed_value=42):
    assert int(train_split * 100) + int(val_split * 100) + int(test_split * 100) == 100, f"Splits must sum to 1, but got {train_split + val_split + test_split}"
    
    seed = Generator().manual_seed(seed_value) # to avoid data leakage
    
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train, val, test = random_split(dataset, [train_size, val_size, test_size], generator=seed)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader