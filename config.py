import transformers
from transformers import AutoConfig

import re

class DefaultConfig:
    MODEL_NAME = "roberta-base"
    MODEL_CONFIG = AutoConfig.from_pretrained(MODEL_NAME)
    SEED = 42
    
    OPTION = ""
    
    TRAIN_BATCH = 32
    VALID_BATCH = 64
    TEST_BATCH = 64
    
    TRAIN_LOG_INTERVAL = 1
    VALID_LOG_INTERVAL = 1