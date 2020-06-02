import math

import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM


class LM:
    def __init__(self, model):
        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model)

        # Model
        self.bertModel = BertForMaskedLM.from_pretrained(model)
        self.bertModel.eval()

    def get_score(self, sentence):
        # Tokenized input
        tokenized_text = self.tokenizer.tokenize(sentence)
        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])

        # Calculate
        predictions = self.bertModel(tokens_tensor)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions.squeeze(), tokens_tensor.squeeze()).data
        return math.exp(loss)
