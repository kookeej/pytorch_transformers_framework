import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel

import gc

from config import DefaultConfig

config = DefaultConfig()

# Customized model
class CustomModel(nn.Module):
    """
    You can customize the model to fit your task.
    """
    def __init__(self, conf):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(config.MODEL_NAME, config=conf)
        """
        Customize!
        """

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids, attention_mask=attention_mask
        )
        """
        Customize!
        """
        
        return outputs