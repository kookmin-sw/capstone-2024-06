import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터 로드 및 전처리
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    X = np.array([eval(vector_str) for vector_str in data.iloc[:, 0]])  # 첫 번째 열을 벡터로 가져오기
    y = data.iloc[:, 2].values.astype(int)  # 마지막 열을 라벨로 가져오기

    # 데이터를 PyTorch 텐서로 변환
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y


# 다층 퍼셉트론(MLP) 모델 정의
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 모델 학습 함수
def train_model(model, criterion, optimizer, X_train, y_train, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# CSV 파일 경로
csv_path = "./data.csv"

# 데이터 로드
X, y = load_data(csv_path)

# 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 파라미터 설정
input_size = X.shape[1]  # 입력 차원은 벡터의 차원과 같습니다.
hidden_size = 100  # 은닉층의 크기
num_classes = len(torch.unique(y))  # 클래스 수

# 모델 생성
model = MLP(input_size, hidden_size, num_classes)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
print("Training started...")
train_model(model, criterion, optimizer, X_train, y_train, num_epochs=300)
print("Training completed!")
# 모델 저장
torch.save(model.state_dict(), './models/model0412')
print("Model saved!")
