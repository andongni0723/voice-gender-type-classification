import src.train as train
import src.model_eval as model_eval
from src.data_manager import DataManager
from src.spec_process import SpectrogramProcessor


def main():
    device = train.device_detected()
    processor = SpectrogramProcessor()
    train_loader, valid_loader = DataManager('data/train', processor, 1000).get_dataloaders(batch_size=32)
    test_loader , _ = DataManager('data/train', processor, 1000).get_dataloaders(batch_size=32, valid_ratio=0)
    classify_model = train.train_with_dataloader(train_loader, valid_loader, epochs=50, lr=1e-4, device=device)
    model_eval.print_confusion_matrix(classify_model, test_loader, device=device)

if __name__ == "__main__":
    main()
