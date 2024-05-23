import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.users, self.items, self.ratings = self.preprocess(ratings)
    
    def preprocess(self, ratings):
        users = torch.tensor(ratings["userId"].values)
        items = torch.tensor(ratings["movieId"].values)
        ratings = torch.tensor(ratings["rating"].values)
        return users, items, ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
    

class FISM(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.query_embeddings = nn.Embedding(num_items, embedding_dim)
        self.target_embeddings = nn.Embedding(num_items, embedding_dim)
    
    def forward(self, user, item_i, item_j):
        user_bias = self.user_bias(user)
        item_i_bias = self.item_bias(item_i)
        item_j_bias = self.item_bias(item_j)

        query_emb = self.query_embeddings(item_j)
        target_emb = self.target_embeddings(item_i)
        pred = torch.bmm(query_emb, target_emb)
