from argparse import ArgumentParser
import wandb
import yaml
import torch
from tqdm import tqdm
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import optuna
from optuna.integration import WeightsAndBiasesCallback
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import gc

from src.utils import load_model, collate_fn, compute_metrics
from src.data import ClipDataset


def create_objective(model_name: str, config: dict, device: torch.device, 
                     train_dataset, val_dataset, pos_weights, bias, logger, study_group_name: str):
    """
    Factory function to create an objective function for Optuna optimization.
    """
    
    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        """
        
        use_attention_pooling = trial.suggest_categorical("attention_pooling", [True, False])
        use_separate_lr = trial.suggest_categorical("use_separate_lr", [True, False])
        
        if use_separate_lr:
            classifier_lr = trial.suggest_float("classifier_lr", 1e-5, 1e-2, log=True)
            lora_lr = trial.suggest_float("lora_lr", 1e-6, 1e-4, log=True)
        else:
            learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        
        num_classifier_layers = trial.suggest_int("num_classifier_layers", 0, 2)
        classifier_dims = []
        for i in range(num_classifier_layers):
            dim = trial.suggest_categorical(f"classifier_dim_{i}", [64, 128, 256])
            classifier_dims.append(dim)
        
        classifier_dropout = trial.suggest_float("classifier_dropout", 0.0, 0.5)
        classifier_activation = trial.suggest_categorical("classifier_activation", ["relu", "gelu"])

        lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
        lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        layer_config_name = trial.suggest_categorical("layer_config", ["basic", "extended"])
        config_map = {
            "basic": ["q_proj", "v_proj"],
            "extended": ["q_proj", "v_proj", "up_proj", "down_proj"]
        }
        lora_layers = config_map[layer_config_name]

        trial_config = {
            "attention_pooling": use_attention_pooling,
            "use_separate_lr": use_separate_lr,
            "weight_decay": weight_decay,
            "classifier_dims": classifier_dims,
            "classifier_dropout": classifier_dropout,
            "classifier_activation": classifier_activation,
            "num_epochs": config.get("num_epochs", 1),
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_layers": lora_layers
        }
        if use_separate_lr:
            trial_config["classifier_lr"] = classifier_lr
            trial_config["lora_lr"] = lora_lr
        else:
            trial_config["learning_rate"] = learning_rate
        
        wandb.init(
            project="vlm-hyperparam-optim",
            group=study_group_name,
            name=f"trial_{trial.number}",
            config=trial_config,
            reinit=True
        )
        
        try:
            model = load_model(model_name, **config.get("model_params", {}))
            model = model.to(device)
            
            classifier_config = {
                "dims": classifier_dims,
                "activation": classifier_activation,
                "use_bias": True,
                "dropout": classifier_dropout
            }
            model.build_classifier(classifier_config=classifier_config, bias=bias)
            
            if use_attention_pooling:
                model.build_attention_pooling()
            
            train_backbone = config.get("train_backbone", False)
            if train_backbone:
                lora_config = config.get("lora_config", {})
                lora_config["lora_r"] = lora_r
                lora_config["lora_alpha"] = lora_alpha
                lora_config["target_modules"] = lora_layers
                model.inject_lora_layers(lora_config=lora_config)
            
            for param in model.parameters():
                param.requires_grad = False
            
            if train_backbone:
                for name, param in model.backbone.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True
            
            if use_attention_pooling:
                for param in model.attn_pool.parameters():
                    param.requires_grad = True
            
            for param in model.classifier.parameters():
                param.requires_grad = True
            
            model.to(device)
            
            if use_separate_lr and train_backbone:
                lora_params = [p for n, p in model.backbone.named_parameters() if "lora_" in n and p.requires_grad]
                classifier_params = list(model.classifier.parameters())
                if use_attention_pooling:
                    classifier_params += list(model.attn_pool.parameters())
                
                param_groups = [
                    {"params": classifier_params, "lr": classifier_lr},
                    {"params": lora_params, "lr": lora_lr}
                ]
                optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
            else:
                lr = learning_rate if not use_separate_lr else classifier_lr
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr,
                    weight_decay=weight_decay
                )
            
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.get("batch_size", 4),
                shuffle=True,
                num_workers=config.get("num_workers", 0),
                pin_memory=config.get("num_workers", 0) > 0,
                collate_fn=collate_fn
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.get("batch_size", 4),
                shuffle=False,
                num_workers=config.get("num_workers", 0),
                pin_memory=config.get("num_workers", 0) > 0,
                collate_fn=collate_fn
            )
            
            N = len(train_dataset)
            N_val = len(val_dataset)
            C = model.num_classes
            num_epochs = config.get("num_epochs", 1)
            
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            scaler = GradScaler(enabled=(amp_dtype == torch.float16))
            
            best_val_loss = float('inf')
            
            for epoch in range(num_epochs):
                logger.info(f"Trial {trial.number} - Epoch {epoch+1}/{num_epochs}")
                model.train()
                
                train_loss = 0.0
                total_samples = 0
                
                for batch in tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}", 
                                  total=N/config.get("batch_size", 4)):
                    
                    labels = batch.pop("labels").to(device)
                    optimizer.zero_grad()
                    
                    with autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(**batch)
                        loss = criterion(logits, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item() * labels.size(0)
                    total_samples += labels.size(0)
                
                train_loss /= total_samples
                
                model.eval()
                val_loss = 0.0
                val_total_samples = 0
                val_logits_tensor = torch.empty((N_val, C), dtype=torch.float32)
                val_labels_tensor = torch.empty((N_val, C), dtype=torch.float32)
                
                with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype):
                    for batch in tqdm(val_loader, desc=f"Trial {trial.number} Validation", 
                                      total=N_val/config.get("batch_size", 4)):
                        
                        labels = batch.pop("labels").to(device)
                        logits = model(**batch)
                        loss = criterion(logits, labels)
                        
                        val_logits_tensor[val_total_samples:val_total_samples + logits.size(0), :] = logits.detach().cpu()
                        val_labels_tensor[val_total_samples:val_total_samples + labels.size(0), :] = labels.detach().cpu()
                        
                        val_loss += loss.item() * labels.size(0)
                        val_total_samples += labels.size(0)
                
                val_loss /= val_total_samples
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                metrics = compute_metrics(val_logits_tensor, val_labels_tensor)
                
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch": epoch + 1,
                    **{f"val_{k}": v for k, v in metrics.items()}
                })
                
                trial.report(val_loss, epoch)
                
                if trial.should_prune():
                    wandb.finish()
                    raise optuna.TrialPruned()
            
            wandb.finish()
            
            del model, optimizer, criterion
            torch.cuda.empty_cache()
            gc.collect()
            
            return best_val_loss
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}")
            wandb.finish()
            raise
    
    return objective


def main():

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to use.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--study_name", type=str, default=None, help="Name of the Optuna study.")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///optuna.db).")

    args = parser.parse_args()
    model_name = args.model
    debug = args.debug
    n_trials = args.n_trials

    with open("configs/hp_optim_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}

    if debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting hyperparameter optimization with model {model_name}")
    logger.debug(f"Configuration: {config}")

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    study_group_name = f"HPOptim_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study_name = args.study_name or study_group_name
    logger.info(f"Study name: {study_name}")

    temp_model = load_model(model_name, **config.get("model_params", {}))
    processor = temp_model.processor
    num_classes = temp_model.num_classes
    del temp_model
    torch.cuda.empty_cache()
    gc.collect()

    prompt = config.get("prompt", "What's happening in the video?")
    system_message = config.get("system_message", "You are a helpful assistant.")
    logger.info(f"Using prompt: {prompt}")
    
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

    logger.info("Loading datasets...")
    train_dataset = ClipDataset(
        video_csv=config.get("train_data", "data/train.csv"),
        prompt_template=prompt_template,
        processor=processor,
        num_frames=config.get("num_frames", None)
    )
    val_dataset = ClipDataset(
        video_csv=config.get("validation_data", "data/validation.csv"),
        prompt_template=prompt_template,
        processor=processor,
        num_frames=config.get("num_frames", None)
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    pos_weights = train_dataset.compute_pos_weights()
    bias = train_dataset.compute_bias()
    logger.info(f"Positive weights for loss: {pos_weights}")
    logger.info(f"Bias for classifier: {bias}")

    objective = create_objective(
        model_name=model_name,
        config=config,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        pos_weights=pos_weights,
        bias=bias,
        logger=logger,
        study_group_name=study_group_name
    )

    file_path = f"optuna_studies/{args.storage}" if args.storage is not None else f"optuna_studies/{study_name}.log"
    storage = JournalStorage(JournalFileBackend(file_path))
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    logger.info(f"Starting optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info("=" * 50)
    logger.info("Optimization completed!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best validation loss: {study.best_trial.value:.6f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    best_params_path = f"configs/best_params_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(best_params_path, "w") as f:
        yaml.dump(study.best_trial.params, f, default_flow_style=False)
    logger.info(f"Best parameters saved to {best_params_path}")
    
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        import plotly.io as pio
        
        fig_history = plot_optimization_history(study)
        pio.write_html(fig_history, f"optuna_history_{study_name}.html")
        
        fig_importance = plot_param_importances(study)
        pio.write_html(fig_importance, f"optuna_importance_{study_name}.html")
        
        logger.info("Optimization plots saved.")
    except ImportError:
        logger.warning("plotly not installed, skipping visualization export.")


if __name__ == "__main__":
    main()