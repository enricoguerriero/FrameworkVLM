from argparse import ArgumentParser
import wandb
import torch
from tqdm import tqdm
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from torch.amp import autocast

from .utils import load_model, collate_fn, compute_metrics
from .clip_dataset import ClipDataset

def main():

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to use.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a pre-trained model checkpoint.")

    args = parser.parse_args()
    debug = args.debug    
    saved_model_path = args.model_path
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    saved_model = torch.load(saved_model_path, map_location=device, weights_only=False)

    config = saved_model.get("config", {})

    if debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logger = logging.getLogger(__name__)
    logger.info(f"Starting testing with model {args.model}")
    logger.debug(f"Configuration: {config}")

    model = load_model(args.model, **config.get("model_params", {}))
    logger.debug(f"Model architecture: {model}")
    model = model.to(device)

    wandb.init(project="vlm-test", name=f"Test_{model.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", config=config)

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

    test_dataset = ClipDataset(
        video_csv=config.get("test_data", "test_data.csv"),
        prompt_template=prompt_template,
        processor=model.processor,
        num_frames=config.get("num_frames", None)
    )
    logger.info(f"Test dataset size: {len(test_dataset)} samples")
    test_loader = DataLoader(
        test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = config.get("num_workers", 0),
        pin_memory = config.get("num_workers", 0) > 0,
        collate_fn = collate_fn
    )
    logger.debug("Test data loaded.")
    model.load_classifier(saved_model)
    logger.info(f"Loaded classifier from {saved_model_path}")
    model.load_backbone(saved_model)
    logger.info(f"Loaded backbone from {saved_model_path}")

    model.eval()
    logits_tensor = torch.empty((len(test_dataset), model.num_classes), dtype=torch.float32)
    labels_tensor = torch.empty((len(test_dataset), model.num_classes), dtype=torch.float32)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing", total=len(test_loader))):
            labels = batch.pop("labels").to(device)

            with autocast(device_type="cuda", dtype=torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16):
                logits = model(**batch)

            logits_tensor[i] = logits.detach().cpu()
            labels_tensor[i] = labels.detach().cpu()
    
    metrics = compute_metrics(logits_tensor, labels_tensor)
    logger.info(f"Test Metrics: {metrics}")
    wandb.log(metrics)
    logger.info("Testing completed.")

if __name__ == "__main__":
    main()