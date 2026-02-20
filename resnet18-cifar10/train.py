import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from codecarbon import track_emissions

api_key = os.environ.get("CODECARBON_API_TOKEN")
experiment_id = os.environ.get("CODECARBON_EXPERIMENT_ID")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 on CIFAR-10"
    )
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes")
    return parser.parse_args()

output_dir = "/app/resnet18-cifar10/emissions"
os.makedirs(output_dir, exist_ok=True)
@track_emissions(
        api_endpoint="https://api.codecarbon.io",
        api_key=api_key,
        experiment_id=experiment_id,
        save_to_api=True,
        log_level="WARNING",
        output_dir="/app/resnet18-cifar10/emissions",
        country_iso_code="CAN"
        )
def train(model, trainloader, criterion, optimizer, scheduler, epochs, device):
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        avg_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")

def main():
    model_dir = "/app/resnet50-cifar10/models"
    os.makedirs(model_dir, exist_ok=True)

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Model
    model = resnet18(num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    train(model, trainloader, criterion, optimizer, scheduler, args.epochs, device)

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    print(f"Test accuracy: {acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), f"{model_dir}/resnet18_cifar10.pt")


if __name__ == "__main__":
    main()

