from src.models import VisionLanguageModel

def load_model(model_name: str, **kwargs) -> VisionLanguageModel:
    """
    Factory function to load a vision-language model by name.
    """
    if model_name == "LLaVANeXT":
        from src.models import LLaVANeXT
        return LLaVANeXT(**kwargs)
    elif model_name == "Qwen3VL":
        from src.models import Qwen3VL
        return Qwen3VL(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not recognized.")