import torch
import torchvision
import numpy as np
import timm
from tqdm import tqdm
import argparse
import os
import time
import cli_utils
import tent
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from codecarbon import track_emissions
from robustbench.data import load_imagenetc

def imagenet_collate_fn(batch):
    images, labels = zip(*batch)
    return {"image": torch.stack(images), "label": torch.tensor(labels, dtype=torch.long)}

def load_pretrained_model(model):
    if args.model == 'resnet50':
        return timm.create_model('resnet50.a1_in1k', pretrained=True)
    elif args.model == 'vit-b': 
    	return timm.create_model('vit_base_patch16_224', pretrained=True)
    else:
        raise ValueError("Model not supported")
    

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adapt with Tent on ImageNet-C"
    )
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for adapting")
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'])
    parser.add_argument('--model', choices=['resnet50, vit-b'], default='resnet50')
    parser.add_argument('--corruption', type=str, default='gaussian_noise')
    parser.add_argument('--level', type=int, choices=[1,2,3,4,5], default=5)
    #parser.add_argument('--adaptation', choices=['online', 'episodic']
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
    for batch in tqdm(loader):
        labels = batch["label"].to(device)
        inputs = batch["image"].to(device)

        outputs = model(inputs)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).float().mean()
        total += labels.size(0)
    acc = correct / total
    print(f"Accuracy on {corruption}: {acc:.4f}")

def main():	
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Capability: {torch.cuda.get_device_capability(0)}')
    print(f"Using device: {device}")
    if args.model == 'resnet50':
        lr = (0.00025 / 64) * args.batch_size * 2 if args.batch_size < 32 else 0.00025
    elif args.model == 'vit-b':
        lr = (0.001 / 64) * args.batch_size
    else:
        assert False, NotImplementedError

    if args.corruption == 'all':
        corruptions=[
         'gaussian_noise', 'shot_noise', 'impulse_noise','defocus_blur',
         'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
         'brightness', 'contrast', 'elastic_transform', 'pixelate',
         'jpeg_compression'
         ]
    else:
        corruptions=[args.corruption]

    for corrupt in corruptions:
        print(f"--- Adapting to {corrupt} ---")

        model = load_pretrained_model(args.model)
        model = model.to(device)
        model = tent.configure_model(model)
        params, param_names = tent.collect_params(model)
        optimizer = torch.optim.Adam(params,lr=lr)
        tented_model = tent.Tent(model, optimizer)
        print(f"Adapting {len(params)} params")

        local_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        root='/app/data/'
        #dataset_path = os.path.join(root, corrupt, str(args.level))
        #dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=local_transform)
        #loader = DataLoader(dataset, batch_size=args.batch_size, 
        #    shuffle=False, num_workers=4, pin_memory=True,
        #    collate_fn=imagenet_collate_fn)
        #model = timm.create_model('resnet50', pretrained=True)
        #categories = model.pretrained_cfg.get('label_names', [])
        x_test, y_test = load_imagenetc(
            n_examples=5000,
            severity=args.level,
            data_dir='/app/data/',
            corruptions=[corrupt]
        )
        dataset = TensorDataset(x_test, y_test)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


        adapt(tented_model, loader, corrupt, device)

if __name__ == "__main__":
    main()

