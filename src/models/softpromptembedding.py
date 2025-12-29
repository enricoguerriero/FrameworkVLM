from torch import nn
import torch

class SoftPromptEmbedding(nn.Module):

    def __init__(self, original_embedding: nn.Embedding, tokenizer, prompt_text, device = 'cuda'):
        super().__init__()
        self.base_embedding = original_embedding
        self.embedding_dim = original_embedding.weight.shape[1]

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        self.num_tokens = len(prompt_ids)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        # batch_size = input_ids.size(0)
        # soft_prompt_expanded = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        # original_embeddings = self.original_embedding(input_ids)
        # return torch.cat([soft_prompt_expanded, original_embeddings], dim=1)

        embeddings = self.original_embedding(input_ids)
        return embeddings