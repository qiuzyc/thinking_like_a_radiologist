import torch
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self, embed_path="embed.ckpt", vocab_size=8192, hidden_size=256, generator_hidden_size=4096):
        super(PerceptualLoss, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        embedding = torch.load(embed_path, map_location="cpu")
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True).to(dtype=torch.bfloat16)
        self.states_to_hidden = nn.Linear(generator_hidden_size, hidden_size, dtype=torch.bfloat16)
        self.criterion = nn.MSELoss()
        
    def forward(self, input_ids, generated_hidden_states):
        assert input_ids.shape[1] == 1024
        assert generated_hidden_states.shape[1] == 1024
        labels = self.embedding(input_ids - 4)  # input ids: (batch, 1024)
        features = self.states_to_hidden(generated_hidden_states)   # hidden states: (batch, 1024, generator_hidden_size)
        return self.criterion(features, labels)