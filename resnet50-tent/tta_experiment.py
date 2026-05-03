import torch
import numpy as np
import timm
import argparse
import os
import time
import cli_utils
import tent
from torchvision import datasets, transforms
from codecarbon import track_emissions
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
        log_level="INFO",
        output_dir=output_dir,
        country_iso_code="CAN",
        region="british columbia",
        tracking_mode='machine',
        measure_power_secs=15,
        )
def adapt(model, loader, corruption, device):
    acc_batch=[]
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        acc = (outputs.max(1)[1] == targets).float().mean()
        acc_batch.append(acc.item())
    acc_mean = np.mean(acc_batch)
    print(f"Accuracy on {corruption}: {acc_mean:.4f}")

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

# Direct loader for local debugging
    local_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ])

    for corrupt in corruptions:
        print(f"--- Adapting to {corrupt} ---")

        model = load_pretrained_model(args.model)
        model = model.to(device)
        model = tent.configure_model(model)
        params, param_names = tent.collect_params(model)
        optimizer = torch.optim.SGD(params, lr, momentum=0.9)
        tented_model = tent.Tent(model, optimizer)

#    x_test, y_test = load_imagenet_c(
#        n_examples=5000,
#        severity=args.severity,
#        data_dir=args.data_dir,
#        corruptions=[corrupt]
#    )


        dataset = datasets.ImageFolder(root=f'./data/{corrupt}/{args.level}', transform=local_transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        adapt(tented_model, loader, corrupt, device)

if __name__ == "__main__":
    main()

