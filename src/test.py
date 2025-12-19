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
    parser.add_argument("--model_path", type=str, default=None, help="Path to a pre-trained model checkpoint.")

    args = parser.parse_args()
    debug = args.debug    
    saved_model_path = args.model_path
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logger = logging.getLogger(__name__)
    logger.info(f"Starting testing with model {args.model}")
    logger.debug(f"Configuration: {config}")

    model = load_model(args.model, **config.get("model_params", {}))
    logger.debug(f"Model architecture: {model}")
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
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
    saved_model = torch.load(saved_model_path, map_location=device)
    model.load_classifier(saved_model)
    logger.info(f"Loaded classifier from {saved_model_path}")
    model.load_backbone(saved_model)
    logger.info(f"Loaded backbone from {saved_model_path}")