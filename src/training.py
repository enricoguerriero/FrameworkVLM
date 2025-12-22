from argparse import ArgumentParser
from xml.parsers.expat import model
import wandb
import yaml
import torch
from tqdm import tqdm
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from .utils import load_model, collate_fn, compute_metrics
from .clip_dataset import ClipDataset

def main():

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to use.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
    parser.add_argument("--only_train", action="store_true", default=False, help="Only run training, skip validation.")

    args = parser.parse_args()
    debug = args.debug
    only_train = args.only_train

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    if debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with model {args.model}")
    logger.debug(f"Configuration: {config}")

    model = load_model(args.model, **config.get("model_params", {}))
    logger.debug(f"Model architecture: {model}")
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    wandb.init(project="vlm-training", name=f"Training_{model.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", config=config)

    prompt = config.get("prompt", "What's happening in the video?")
    system_message = config.get("system_message", "You are a helpful assistant.")
    logger.info(f"Using prompt: {prompt}")
    logger.info(f"Using system message: {system_message}")

    prompt_template = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"}
            ]
        }
    ]

    train_dataset = ClipDataset(
        video_csv = config.get("train_data", "train_data.csv"),
        prompt_template = prompt_template,
        processor = model.processor,
        num_frames = config.get("num_frames", None)
    )
    val_dataset = ClipDataset(
        video_csv = config.get("validation_data", "val_data.csv"),
        prompt_template = prompt_template,
        processor = model.processor,
        num_frames = config.get("num_frames", None)
    )

    if only_train:
        logger.info("Only training mode enabled; skipping validation.")
        old_train_dataset = train_dataset
        train_dataset = old_train_dataset + val_dataset
        logger.info(f"Combined training dataset size: {len(train_dataset)}")
    else:
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    logger.debug("Starting data loading ...")
    train_loader = DataLoader(
        train_dataset,
        batch_size = config.get("batch_size", 4),
        shuffle = True,
        num_workers = config.get("num_workers", 0),
        pin_memory = config.get("num_workers", 0) > 0,
        collate_fn = collate_fn
    )
    logger.debug("Training data loaded.")
    if not only_train:
        val_loader = DataLoader(
            val_dataset,
            batch_size = config.get("batch_size", 4),
            shuffle = False,
            num_workers = config.get("num_workers", 0),
            pin_memory = config.get("num_workers", 0) > 0,
            collate_fn = collate_fn
        )
        logger.debug("Validation data loaded.")
    
    if only_train:
        count1, n1 = old_train_dataset._get_label_counts(old_train_dataset.data)
        count2, n2 = val_dataset._get_label_counts(val_dataset.data)
        total_counts = count1 + count2
        total_n = n1 + n2
        pos_weights = (total_n - total_counts) / (total_counts + 1e-6)
        bias = torch.log(total_counts / (total_n - total_counts + 1e-6))
    else:
        pos_weights = train_dataset.compute_pos_weights()
        bias = train_dataset.compute_bias()
    logger.info(f"Positive weights for loss: {pos_weights}")
    logger.info(f"Bias for classifier: {bias}")

    model.build_classifier(
        classifier_config = config.get("classifier_config", {}),
        bias = bias
    )
    logger.debug(f"Classifier architecture: {model.classifier}")

    if config.get("train_backbone", False):
        model.inject_lora_layers(
            lora_config = config.get("lora_config", {})
        )
        logger.debug("LoRA layers injected.")
        logger.debug(f"Full model architecture after LoRA injection: {model}")

    for param in model.parameters():
        param.requires_grad = False
    if config.get("train_backbone", False):
        for name, param in model.backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params}")
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters in model: {all_params}")

    model.to(device)
    logger.debug("Model moved to device.")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config.get("learning_rate", 1e-5),
        weight_decay = config.get("weight_decay", 0.001)
    )
    logger.debug("Optimizer initialized.")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    logger.debug("Loss function initialized.")

    N = len(train_dataset)
    if not only_train:
        N_val = len(val_dataset)
    C = model.num_classes

    val_step = config.get("validation_step", None) if not only_train else None
    num_epochs = config.get("num_epochs", 1)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.debug(f"Using AMP dtype: {amp_dtype}")
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))

    for epoch in range(num_epochs):

        logger.info(f"Starting epoch {epoch+1}/{num_epochs} ...")
        model.train()

        train_loss =0.0
        total_samples = 0
        logits_tensor = torch.empty((N, C), dtype=torch.float32)
        labels_tensor = torch.empty((N, C), dtype=torch.float32)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", total=N/config.get("batch_size", 4)):

            labels = batch.pop("labels").to(device)
            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(**batch)
                loss = criterion(logits, labels)
            
            logits_tensor[total_samples:total_samples + logits.size(0), :] = logits.detach().cpu()
            labels_tensor[total_samples:total_samples + labels.size(0), :] = labels.detach().cpu()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            if total_samples % 50 == 0:
                wandb.log({
                    "Train Loss": train_loss / total_samples,
                    "Epoch": epoch + 1,
                    "Batch": total_samples
                })

            if val_step is not None and total_samples % val_step == 0:
                model.eval()

                val_loss = 0.0
                val_total_samples = 0
                val_logits_tensor = torch.empty((N_val, C), dtype=torch.float32)
                val_labels_tensor = torch.empty((N_val, C), dtype=torch.float32)

                with torch.no_grad(), autocast(device_type="cuda"):
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", total=N_val/config.get("batch_size", 4)):

                        labels = batch.pop("labels").to(device)
                        logits = model(**batch)
                        loss = criterion(logits, labels)

                        val_logits_tensor[val_total_samples:val_total_samples + logits.size(0), :] = logits.detach().cpu()
                        val_labels_tensor[val_total_samples:val_total_samples + labels.size(0), :] = labels.detach().cpu()

                        val_loss += loss.item() * labels.size(0)
                        val_total_samples += labels.size(0)

                val_loss /= val_total_samples
                wandb.log({
                    "Validation Loss": val_loss,
                    "Epoch": epoch + 1,
                    "Batch": val_total_samples
                })

                metrics = compute_metrics(val_logits_tensor, val_labels_tensor)
                metrics["epoch"] = epoch + 1
                metrics["batch"] = val_total_samples
                wandb.log(metrics)

                model.train()
                save_path = f"{config.get("checkpoint_path", "checkpoints/")}_{model.model_name}_epoch{epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "backbone": model.backbone.state_dict(),
                    "classifier": model.classifier.state_dict(),
                    "attention_pooling": model.attn_pool.state_dict() if hasattr(model, "attn_pool") else None,
                    "processor": model.processor,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss if val_step else None,
                    "metrics": metrics,
                    "lora_config": config.get("lora_config", {}),
                    "classifier_config": config.get("classifier_config", {}),
                    "config": config
                }, save_path)

        train_loss /= total_samples
        wandb.log({
            "Train Loss Epoch": train_loss,
            "Epoch": epoch + 1
        })

        if not only_train:
            model.eval()

            val_loss = 0.0
            val_total_samples = 0
            val_logits_tensor = torch.empty((N_val, C), dtype=torch.float32)
            val_labels_tensor = torch.empty((N_val, C), dtype=torch.float32)

            with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype):
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Full Validation", total=N_val/config.get("batch_size", 4)):

                    labels = batch.pop("labels").to(device)
                    logits = model(**batch)
                    loss = criterion(logits, labels)

                    val_logits_tensor[val_total_samples:val_total_samples + logits.size(0), :] = logits.detach().cpu()
                    val_labels_tensor[val_total_samples:val_total_samples + labels.size(0), :] = labels.detach().cpu()

                    val_loss += loss.item() * labels.size(0)
                    val_total_samples += labels.size(0)

            val_loss /= val_total_samples
            wandb.log({
                "Validation Loss Epoch": val_loss,
                "Epoch": epoch + 1
            })

            metrics = compute_metrics(val_logits_tensor, val_labels_tensor)
            metrics["epoch"] = epoch + 1
            wandb.log(metrics)

        metrics = compute_metrics(logits_tensor, labels_tensor)
        metrics["epoch"] = epoch + 1
        wandb.log({"Train Epoch Metrics": metrics})
        
        logger.info(f"Epoch {epoch+1} completed. Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}" if not only_train else f"Epoch {epoch+1} completed. Train Loss: {train_loss:.4f}")
        save_path = f"{config.get('checkpoint_path', 'checkpoints/')}_{model.model_name}_epoch{epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" + ".pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "backbone": model.backbone.state_dict(),
            "classifier": model.classifier.state_dict(),
            "attention_pooling": model.attn_pool.state_dict() if hasattr(model, "attn_pool") else None,
            "processor": model.processor,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss if not only_train else None,
            "metrics": metrics,
            "lora_config": config.get("lora_config", {}),
            "classifier_config": config.get("classifier_config", {}),
            "config": config
        }, save_path)

        logger.info(f"Model checkpoint saved at {save_path}")
    
    logger.info("Training completed.")
    save_path = f"{config.get('save_path', 'models/')}_{model.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "backbone": model.backbone.state_dict(),
        "classifier": model.classifier.state_dict(),
        "processor": model.processor,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": num_epochs,
        "train_loss": train_loss,
        "val_loss": val_loss if not only_train else None,
        "metrics": metrics,
        "lora_config": config.get("lora_config", {}),
        "classifier_config": config.get("classifier_config", {}),
        "config": config
    }, save_path)
    logger.info(f"Final model saved at {save_path}")

if __name__ == "__main__":
    main()