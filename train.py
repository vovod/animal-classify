import torch.nn as nn
import torch
import os
import time
from tqdm import tqdm
from torchvision import transforms, datasets, models

def train_model():
    data_transforms = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if not os.path.exists('weights/'):
        os.makedirs('weights/')

    dataset_dir = "E:\\animal-classify\\animals\\animals"

    bsz = 2

    # Split data
    train_dataset = datasets.ImageFolder(root=dataset_dir, transform=data_transforms)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Load data
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bsz, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=bsz, shuffle=False, num_workers=4)

    load_dict = "weights/"

    TRAIN_MODE = {"anms": 90}

    model_ft = models.efficientnet_v2_s(num_classes=TRAIN_MODE.get("anms")).to(device)
    model_ft.classifier[1] = nn.Linear(model_ft.classifier[1].in_features, TRAIN_MODE.get("anms")).to(device)

    losses = []
    accuracies = []
    epochs = 200
    start = time.time()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_ft.parameters(), lr=0.01)
    try:
        for epoch in range(epochs):
            model_ft.train()
            epoch_loss = 0
            epoch_accuracy = 0
            for X, y in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                preds = model_ft(X)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()
                accuracy = (preds.argmax(dim=1) == y).float().mean()
                epoch_accuracy += accuracy.item()
                epoch_loss += loss.item()
            epoch_accuracy /= len(train_dataloader)
            epoch_loss /= len(train_dataloader)
            accuracies.append(epoch_accuracy)
            losses.append(epoch_loss)
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, Time: {time.time() - start:.2f}s")
            model_ft.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                test_epoch_accuracy = 0
                for test_X, test_y in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                    test_X = test_X.to(device)
                    test_y = test_y.to(device)

                    test_preds = model_ft(test_X)
                    test_loss = loss_fn(test_preds, test_y)

                    test_epoch_loss += test_loss.item()
                    test_accuracy = (test_preds.argmax(dim=1) == test_y).float().mean()
                    test_epoch_accuracy += test_accuracy.item()
                test_epoch_accuracy /= len(val_dataloader)
                test_epoch_loss /= len(val_dataloader)
                print(f"Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_epoch_accuracy:.4f}, Time: {time.time() - start:.2f}s")
                torch.save(model_ft.state_dict(), os.path.join(
                    load_dict, f"Epoch{epoch+1}_Acc{test_epoch_accuracy:.4f}.pth"))
    except Exception as e:
        print(f"Training stopped due to an error: {e}")

if __name__ == "__main__":
    train_model()
