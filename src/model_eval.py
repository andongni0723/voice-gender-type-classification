import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


def print_confusion_matrix(model: nn.Module, test_loader: DataLoader, device: str) -> None:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            p = model(bx.to(device)).argmax(1).cpu()
            y_true += by.tolist()
            y_pred += p.tolist()
    print(classification_report(y_true, y_pred, target_names=['male', 'female']))
    print(confusion_matrix(y_true, y_pred))