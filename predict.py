import numpy as np
import sys
from utils.common import load_class
import importlib

model_config = {
    "model": {
        "class": "models.CP_ResNet",
        "params": {
            "depth": 26,
            "base_channels": 128,
            "n_blocks_per_stage": [
                3,
                1,
                1
            ],
            "multi_label": False,
            "prediction_threshold": 0.4,
            "stage1": {
                "maxpool": [
                    1,
                    2
                ],
                "k1s": [
                    3,
                    3,
                    3
                ],
                "k2s": [
                    1,
                    3,
                    3
                ]
            },
            "stage2": {
                "maxpool": [
                    1
                ],
                "k1s": [
                    3
                ],
                "k2s": [
                    3
                ]
            },
            "stage3": {
                "maxpool": [],
                "k1s": [
                    3
                ],
                "k2s": [
                    1
                ]
            },
            "block_type": "basic",
            "use_bn": True
        }
    }
}

logits = []

assert len(sys.argv[2:]) > 1

model = getattr(importlib.import_module(model_config['class']), 'get_model')(
    input_shape=self.data_set.get_input_shape(),
    output_shape=self.data_set.get_output_shape(),
    **config['model'].get('params', {})
)

for m in sys.argv[2:]:

