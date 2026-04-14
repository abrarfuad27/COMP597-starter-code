import src.config as config # Configurations
import src.trainer as trainer # Trainer base class
from src.models.convnext_large.model import convnext_large_init

from typing import Any, Dict, Optional, Tuple
import torch.utils.data as data

model_name = "convnext_large"

def init_model(conf : config.Config, dataset : data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    return convnext_large_init(conf, dataset)