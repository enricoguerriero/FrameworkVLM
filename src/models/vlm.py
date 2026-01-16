import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from .classifier import ClassifierHead
from .attentionpooling import AttentionPooling

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
        self.attn_pool = None
        self.input_device = None

    def pooling(self, x, mask):
        
        if self.attn_pool is not None:
            # padding_mask = (mask != self.backbone.config.pad_token_id)
            return self.attn_pool(x, mask)
        
        pooled = (x * mask.unsqueeze(-1)).sum(1) / \
                mask.sum(1, keepdim=True).clamp(min=1)
        return pooled

    def forward(self, pixel_values_videos: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):

        outputs = self.backbone(
            pixel_values_videos=pixel_values_videos.to(self.input_device),
            input_ids=input_ids.to(self.input_device),
            attention_mask=attention_mask.to(self.input_device),
            return_dict=True,
            output_hidden_states=True,
        )

        h = outputs.hidden_states[-1]
        
        if self.attn_pool:
            mask = attention_mask.bool().to(self.input_device)
        else:
            mask = (input_ids == self.backbone.config.video_token_index).to(self.input_device)
        pooled = self.pooling(h, mask)
               
        logits = self.classifier(pooled.float())

        return logits
    
    def load_backbone(self, checkpoint: dict, config: dict = None):

        lora_config = config.get("lora_config", None)
        self.inject_lora_layers(lora_config)
        self.backbone.load_state_dict(checkpoint["backbone"], strict=False)
    
    def load_classifier(self, checkpoint: dict, config: dict = None):

        classifier_config = config.get("classifier_config", None)
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
            task_type = TaskType.FEATURE_EXTRACTION,
            bias = lora_config.get("bias", "none"),
        )

        if modality == "language":
            self.backbone.language_model = get_peft_model(self.backbone.language_model, lora_cfg)
        elif modality == "total":
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
    
    def build_attention_pooling(self):
        self.attn_pool = AttentionPooling(self.hidden_size)

    def load_attention_pooling(self, checkpoint: dict):
        self.build_attention_pooling()
        self.attn_pool.load_state_dict(checkpoint["attn_pool"], strict=False)
        self.attn_pool = self.attn_pool.to(self.input_device)
    