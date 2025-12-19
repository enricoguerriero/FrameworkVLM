from .vlm import VisionLanguageModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

class Qwen3VL(VisionLanguageModel):

    def __init__(self, device: str = "cuda", num_classes: int = 3, backbone_id: str = "Qwen/Qwen3-VL-8B-Instruct"):
        super().__init__(
                    num_classes=num_classes, 
                    backbone_id=backbone_id, 
                    device=device
                    )
        self.model_name = "Qwen3VL"
        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(backbone_id)
        self.processor = AutoProcessor.from_pretrained(backbone_id)
        self.hidden_size = self.backbone.config.text_config.hidden_size
        try:
            self.input_device = torch.device(device)
        except:
            self.input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_tokens_allowed = True

    def forward(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):

        outputs = self.backbone(
            pixel_values_videos=pixel_values_videos.to(self.input_device),
            video_grid_thw=video_grid_thw.to(self.input_device),
            input_ids=input_ids.to(self.input_device),
            attention_mask=attention_mask.to(self.input_device),
            return_dict=True,
            output_hidden_states=True,
        )

        h = outputs.hidden_states[-1]
        video_token_id = self.backbone.config.video_token_id
        
        video_mask = (input_ids == video_token_id).to(h.device)
        
        pooled = (h * video_mask.unsqueeze(-1)).sum(1) / \
               video_mask.sum(1, keepdim=True).clamp(min=1) # Eventually attention pooling here 
               
        logits = self.classifier(pooled.float())

        return logits