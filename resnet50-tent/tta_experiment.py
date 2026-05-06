import torch
import torchvision
import numpy as np
import timm
import argparse
import copy
import os
import csv
import time
import tent
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from codecarbon import track_emissions
from robustbench.data import load_imagenetc

def imagenet_collate_fn(batch):
    images, labels = zip(*batch)
    return {"image": torch.stack(images), "label": torch.tensor(labels, dtype=torch.long)}


def parse_args():
    parser = argparse.ArgumentParser(description="Adapt with Tent on ImageNet-C")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for adapting")
    parser.add_argument('--level', type=int, choices=[1,2,3,4,5], default=5,
                        help="severity level of imagenet-c corruptions")
    parser.add_argument('--method', choices=['continual', 'episodic'], default='episodic',
                        help="episodic resets model and optimizer after each corruption")
    return parser.parse_args()

args = parse_args()
output_dir = "./emissions"
os.makedirs(output_dir, exist_ok=True)

api_key = os.environ.get("CODECARBON_API_TOKEN")
experiment_id = os.environ.get("CODECARBON_EXPERIMENT_ID")
save_to_api = all([api_key, experiment_id])
@track_emissions(
        save_to_api=save_to_api,
        api_key=api_key,
        experiment_id=experiment_id,
        save_to_file=True,
        log_level="ERROR",
        output_dir=output_dir,
        country_iso_code="CAN",
        region="british columbia",
        tracking_mode='machine',
        measure_power_secs=15,
        )
def adapt(model, loader, corruption, device):
    correct, total = 0, 0
    model.eval()
    for inputs, labels in tqdm(loader, desc=f"Adapting to {corruption}"):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / total
    print(f"Accuracy on {corruption}: {acc:.4f}")
    return acc

def main():	
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    lr = (0.00025 / 64) * args.batch_size * 2 if args.batch_size < 32 else 0.00025

    corruptions = 
      [
         'gaussian_noise', 'shot_noise', 'impulse_noise','defocus_blur',
         'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
         'brightness', 'contrast', 'elastic_transform', 'pixelate',
         'jpeg_compression'
      ]

    model = timm.create_model('resnet50.a1_in1k', pretrained=True).to(device)
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    base = copy.deepcopy(model.state_dict())
    print(f"Adapting {len(params)} params")

    results = f"results/{args.method}_level{args.level}.csv"
    with open(results, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not os.path.isfile(results):
            writer.writerow(["corruption", "severity", "accuracy", "batch_size", "lr"])

    for corrupt in corruptions:
        print(f"--- Adapting to {corrupt} ---")
        optimizer = torch.optim.Adam(params,lr=lr)
        tented_model = tent.Tent(model, optimizer)

        root='/app/data/'
        x_test, y_test = load_imagenetc(
            n_examples=5000,
            severity=args.level,
            data_dir='/app/data/',
            corruptions=[corrupt]
        )
        dataset = TensorDataset(x_test, y_test)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        acc = adapt(tented_model, loader, corrupt, device)
        writer.writerow([corrupt, args.level, f"{acc:.4f}", args.batch_size, lr])
        f.flush()
        print(f"Saved {corrupt} results to {results}")

        if args.method == 'episodic':
            model.load_state_dict(base)

if __name__ == "__main__":
    main()

