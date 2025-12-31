import torch
import torch.nn as nn
from transformers import DistilBertModel

class TextEncoder(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', embedding_dim=128, freeze_layers=4, disabled=False):
        super(TextEncoder, self).__init__()
        
        if disabled or model_name is None or model_name.lower() == 'none':
            self.bert = None
            self.hidden_size = 0
            return
            
        self.bert = DistilBertModel.from_pretrained(model_name)
        
        # Freeze first N layers
        if freeze_layers > 0:
            # Freeze embeddings
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            
            # Freeze encoder layers
            for i in range(min(freeze_layers, len(self.bert.transformer.layer))):
                for param in self.bert.transformer.layer[i].parameters():
                    param.requires_grad = False
        
        # Projection to embedding_dim
        self.projection = nn.Linear(self.bert.config.hidden_size, embedding_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token (index 0)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_output)

class IDEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(IDEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

class FusionLayer(nn.Module):
    def __init__(self, text_dim, id_dim, output_dim):
        super(FusionLayer, self).__init__()
        input_dim = text_dim + id_dim
        self.fc = nn.Linear(input_dim, output_dim * 2) # *2 for GLU
        self.glu = nn.GLU()

    def forward(self, text_vec, id_vec):
        combined = torch.cat([text_vec, id_vec], dim=-1)
        return self.glu(self.fc(combined))
