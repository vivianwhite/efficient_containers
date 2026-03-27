import torch
import argparse
import os
import time
import tqdm
import functools
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import BertForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from codecarbon import track_emissions

dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=128)
all_cols = dataset["train"].column_names
remove_cols = [col for col in all_cols if col not in ['label', 'labels']]
tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=remove_cols
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune Bert on GLUE (SST2)"
    )
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5)
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

def train(model, train_loader, val_loader, optimizer, epochs, device):
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                # get batched data
                batch = {k: v.to(device) for k, v in batch.items()}

                # forward pass
                outputs = model(**batch)
                loss = outputs.loss
                if loss is None:
                    print(f"Keys in batch: {batch.keys()}")
                    raise ValueError("Loss is still None! Check batch keys above.")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update progress bar
                pbar.set_postfix(loss=float(loss))
        validate(model, val_loader, device)

def validate(model, val_loader, device):
    model.eval()
    num_matches, num_samples = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            num_matches += (predictions == batch["labels"]).sum().item()
            num_samples += len(batch["labels"])
    print(f"Accuracy: {num_matches / num_samples:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


train_loader = torch.utils.data.DataLoader(
        tokenized_datasets["train"], 
        batch_size=args.batch_size,
        shuffle=True, 
        collate_fn=data_collator
)
val_loader = torch.utils.data.DataLoader(
        tokenized_datasets["validation"], 
        batch_size=args.batch_size,
        shuffle=False, 
        collate_fn=data_collator
)

model_dir = "./models"
os.makedirs(model_dir, exist_ok=True)
train(model, train_loader, val_loader, optimizer, args.epochs, device)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
