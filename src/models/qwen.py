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
        # try:
        #     norm_layer = self.backbone.model.language_model.norm # if lora applied to language model / no lora
        # except:
        #     norm_layer = self.backbone.model.model.language_model.norm # if lora applied to the whole model
        # h_norm = norm_layer(h)
        if self.attn_pool:
            mask = attention_mask.bool().to(self.input_device)
        else:
            mask = (input_ids == self.backbone.config.video_token_id).to(self.input_device)
        pooled = self.pooling(h, mask)
               
        logits = self.classifier(pooled.float())

        return logits
    
    def forward_backbone(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):

        outputs = self.backbone(
            pixel_values_videos=pixel_values_videos.to(self.input_device),
            video_grid_thw=video_grid_thw.to(self.input_device),
            input_ids=input_ids.to(self.input_device),
            attention_mask=attention_mask.to(self.input_device),
            return_dict=True,
            output_hidden_states=True,
        )

        h = outputs.hidden_states[-1]
        norm_layer = self.backbone.model.language_model.norm
        h_norm = norm_layer(h)

        return h_norm, attention_mask

    def forward_classifier(self, features: torch.Tensor, attention_mask: torch.Tensor):
        pooled = self.pooling(features, attention_mask)
        logits = self.classifier(pooled.to(self.input_device))
        return logits
    