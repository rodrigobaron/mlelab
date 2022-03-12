import torch.nn as nn
from transformers import (
    BertPreTrainedModel,
    BertModel,
)

class BertForSentimentClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls_layer = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Inputs:
                -input_ids : Tensor of shape [B, T] containing token ids of sequences
                -attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
                (where B is the batch size and T is the input length)
        """
        # Feed input to BERT and obtain outputs.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain representations of [CLS] heads.
        cls_reps = outputs.last_hidden_state[:, 0]
        # Put these representations to classification layer to obtain logits.
        logits = self.cls_layer(cls_reps)
        # Return logits.
        return logits
