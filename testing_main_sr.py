from testfolder.superresolution import super_resolver
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

super_resolver(
    input_dir="./data/train_images_subset",
    output_dir="./testfolder/superresolved_images",
    model_20m_path="./testfolder/L2A20M.pt",
    model_60m_path="./testfolder/L2A60M.pt",
    device=device
)