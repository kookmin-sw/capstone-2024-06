import torch
import torch.nn as nn


class FISM_simple(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, item_i, item_j, batch_score, n):
        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item_i)

        query_emb = self.item_embeddings(item_j)
        target_emb = self.item_embeddings(item_i).unsqueeze(2)
        batch_score = batch_score.unsqueeze(1)

        sim_mat = torch.bmm(query_emb, target_emb)
        pred = torch.bmm(batch_score, sim_mat).squeeze(-1) / (n - 1)
        pred += self.global_bias + user_bias + item_bias
        return pred.squeeze(-1)
    
    def add_items(self, new_item_embedding, new_item_bias):
        new_embeddings = torch.cat((self.item_embeddings.weight.data, new_item_embedding), 0)
        self.item_embeddings = nn.Embedding.from_pretrained(new_embeddings, freeze=False)

        new_bias = torch.cat((self.item_bias.weight.data, new_item_bias), 0)
        self.item_bias = nn.Embedding.from_pretrained(new_bias, freeze=False)