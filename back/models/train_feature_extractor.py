import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def train(train_loader, pretrained=None, save_path="./model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pretrained is None:
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier = model.classifier[:-1]
    else:
        model = models.vgg16()
        model.classifier = model.classifier[:-1]
        model.load_state_dict(torch.load(pretrained))
    
    model.train()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, vectors in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, vectors)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
    
    torch.save(model.state_dict(), save_path)
