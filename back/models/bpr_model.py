import torch
import torch.nn as nn


class BPRModel(nn.Module):
    def __init__(self, num_users, num_items, design_vectors, dim=50):
        super(BPRModel, self).__init__()
        self.design_embedding = nn.Embedding.from_pretrained(torch.tensor(design_vectors), freeze=True)
        self.linear = nn.Sequential(
            nn.Linear(design_vectors.shape[1], 500),
            nn.ReLU(),
            nn.Linear(500, dim)
        )
        self.item_embedding = nn.Embedding(num_items, dim)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, design_indices, item_indices):
        design_embedding = self.design_embedding(design_indices)
        design_embedding = self.linear(design_embedding)
        item_embedding = self.item_embedding(item_indices)
        return (design_embedding * item_embedding).sum(dim=1)