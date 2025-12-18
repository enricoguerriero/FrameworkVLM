from .vlm import VisionLanguageModel
from transformers import LlavaNextVideoForConditionalGeneration, AutoProcessor
import torch

class LLaVANeXT(VisionLanguageModel):

    def __init__(self, num_classes: int = 3, backbone_id: str = "llava-hf/LLaVA-NeXT-Video-7B-hf", device = None, ds = False):
        super().__init__(num_classes=num_classes, backbone_id=backbone_id, device=device)

        self.model_name = "LLaVANeXT"
        self.backbone = LlavaNextVideoForConditionalGeneration.from_pretrained(backbone_id)
        self.processor = AutoProcessor.from_pretrained(backbone_id, use_fast=True)
        self.processor.tokenizer.padding_side = "left" # Recommended from LLaVA NeXT documentation
        self.hidden_size = self.backbone.config.text_config.hidden_size
        try:
            self.input_device = torch.device(device)
        except:
            self.input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_tokens_allowed = True