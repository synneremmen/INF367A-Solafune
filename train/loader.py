import sys
sys.path.append('../../..')
from torch.utils.data import DataLoader, random_split
from torch import Generator

def get_loader(dataset, batch_size):
    seed = Generator().manual_seed(42)
    train, val_test = random_split(dataset, [0.7, 0.4], seed)
    val, test = random_split(val_test, [0.6, 0.4], seed)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader