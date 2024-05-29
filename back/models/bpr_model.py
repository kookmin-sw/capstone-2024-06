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


def load_model(n_designs, n_items, feature_mat):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPRModel(n_designs, n_items, feature_mat)
    model.load_state_dict(torch.load('models/BPR.pth', map_location=device))
    model.to(device)
    return model


def recommend_items(model, design_idx, num_items, top_n=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    design_tensor = torch.tensor([design_idx])
    item_indices = torch.arange(num_items)

    design_tensor = design_tensor.to(device)
    item_indices = item_indices.to(device)

    with torch.no_grad():
        scores = model(design_tensor, item_indices)

    _, top_indices = torch.topk(scores, top_n)
    top_indices = top_indices.squeeze().cpu().numpy().tolist()

    return top_indices