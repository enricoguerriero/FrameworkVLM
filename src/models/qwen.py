from .vlm import VisionLanguageModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

class Qwen3VL(VisionLanguageModel):

    def __init__(self, device: str = "cuda", num_classes: int = 3, backbone_id: str = "Qwen/Qwen3-VL-8B-Instruct"):
        super().__init__(device, num_classes, backbone_id)

        self.model_name = "Qwen3VL"
        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(backbone_id)
        self.processor = AutoProcessor.from_pretrained(backbone_id)
        self.hidden_size = self.backbone.config.text_config.hidden_size
        try:
            self.input_device = torch.device(device)
        except:
            self.input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_tokens_allowed = True