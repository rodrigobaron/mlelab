import torch
import yaml
from yaml.loader import SafeLoader


def get_accuracy_from_logits(logits, labels):

    probabilties = torch.sigmoid(logits.unsqueeze(-1))
    predictions = (probabilties > 0.5).long().squeeze()
    return (predictions == labels).float().mean()


def read_yaml(file_path: str):
    with open(file_path) as f:
        return yaml.load(f, Loader=SafeLoader)
    