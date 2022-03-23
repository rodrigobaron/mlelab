import torch.nn as nn
from transformers import (
    DistilBertPreTrainedModel,
    DistilBertModel,
)


class DistilBertForSentimentClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.cls_layer = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Inputs:
                -input_ids : Tensor of shape [B, T] containing token ids of sequences
                -attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
                (where B is the batch size and T is the input length)
        """
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_reps = outputs.last_hidden_state[:, 0]
        logits = self.cls_layer(cls_reps)
        return logits
