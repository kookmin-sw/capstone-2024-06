import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# 데이터 로드 함수 정의
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    X = np.array([eval(vector_str) for vector_str in data.iloc[:, 0]])  # 첫 번째 열을 벡터로 가져오기
    y = data.iloc[:, 2].values.astype(int)  # 마지막 열을 라벨로 가져오기

    # 데이터를 PyTorch 텐서로 변환
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y

# 모델 정의
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


# 데이터 로드
X, y = load_data('./data.csv')

# 모델 불러오기
input_size = X.shape[1]  # 입력 차원은 벡터의 차원과 같습니다.
hidden_size = 100  # 은닉층의 크기
num_classes = len(torch.unique(y))  # 클래스 수

model = MLP(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('./models/model0412'))

# 마지막 레이어 제거
model = nn.Sequential(*list(model.children())[:-1])

# 기존 벡터들 변환
transformed_vectors = model(X).detach().numpy()

# 새로운 CSV 파일에 저장 (기존 데이터와 라벨 정보 함께 저장)
new_data = pd.DataFrame(transformed_vectors)
new_data['label'] = y.numpy()
new_data.to_csv('./transformed_data_with_labels.csv', index=False)

print("Transformed data with labels saved!")
