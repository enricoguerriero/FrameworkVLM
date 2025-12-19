import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from .classifier import ClassifierHead


class VisionLanguageModel(nn.Module):
    """
    Generic model class. For future implementations, implement subclasses of this class.
    """

    def __init__(self, num_classes: int = 3, backbone_id: str = None, device = None):
        super().__init__()
        self.device = device
        self.model_name = "VisionLanguageModel"
        self.num_classes = num_classes
        self.backbone_id = backbone_id
        self.video_tokens_allowed = True
        self.backbone = None
        self.processor = None
        self.hidden_size = None
        # self.attn_pool = None
        self.input_device = None

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
        video_token_id = self.backbone.config.video_token_index
        
        video_mask = (input_ids == video_token_id).to(h.device)
        
        pooled = (h * video_mask.unsqueeze(-1)).sum(1) / \
               video_mask.sum(1, keepdim=True).clamp(min=1) # Eventually attention pooling here 
               
        logits = self.classifier(pooled.float())

        return logits
    
    def load_backbone(self, checkpoint: dict):

        lora_config = checkpoint.get("lora_config", None)
        self.inject_lora_layers(lora_config)
        self.backbone.load_state_dict(checkpoint["backbone"], strict=False)
    
    def load_classifier(self, checkpoint: dict):

        classifier_config = checkpoint.get("classifier_config", None)
        self.build_classifier(classifier_config)
        self.classifier.load_state_dict(checkpoint["classifier"], strict=False)
        self.classifier = self.classifier.to(self.input_device)
    
    def inject_lora_layers(self, lora_config: dict):

        modality = lora_config.get("modality")
        lora_cfg = LoraConfig(
            r = lora_config["lora_r"],
            lora_alpha = lora_config["lora_alpha"],
            lora_dropout = lora_config["lora_dropout"],
            target_modules = lora_config["target_modules"],
            task_type = TaskType.CAUSAL_LM,
            bias = lora_config.get("bias", "none"),
        )

        if lora_config["modality"] == "language":
            self.backbone.language_model = get_peft_model(self.backbone.language_model, lora_cfg)
        elif lora_config["modality"] == "total":
            self.backbone = get_peft_model(self.backbone, lora_cfg)
        else:
            raise ValueError("Modality must be either 'language' or 'total'.")
    
    def build_classifier(self, classifier_config: dict, bias = None):

        self.classifier = ClassifierHead(
            in_dim = self.hidden_size,
            dims = classifier_config.get("dims", []),
            num_classes = self.num_classes,
            activation = classifier_config.get("activation", "relu"),
            dropout = classifier_config.get("dropout", 0.2),
            bias = bias if classifier_config.get("use_bias", True) else None,
        )
