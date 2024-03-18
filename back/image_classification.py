import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 데이터 폴더 경로
data_dir = "./train_vector"

# 데이터와 레이블 초기화
data = []
labels = []

# 각 클래스 폴더를 반복하면서 벡터 파일을 로드하고 레이블 지정
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    for file_name in os.listdir(class_dir):
        vector_path = os.path.join(class_dir, file_name)
        # 벡터 파일 로드
        vector = np.load(vector_path)
        data.append(vector)
        labels.append(class_name)

# 데이터와 레이블을 넘파이 배열로 변환
data = np.array(data)
labels = np.array(labels)

# 레이블 인코딩
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 클래스 개수 계산
num_classes = len(label_encoder.classes_)

# 데이터를 학습 세트와 테스트 세트로 분할
train_data, test_data, train_labels, test_labels = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# 레이블을 원-핫 인코딩
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

# 새로운 모델 생성
model = Sequential([
    Dense(512, activation='relu', input_shape=(train_data.shape[1],)),  # 입력 벡터의 크기를 지정합니다.
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(train_data, train_labels, epochs=300, batch_size=32, validation_split=0.1)

# 모델 평가
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
class_names = label_encoder.classes_
print("Class Names:", class_names)
# 모델 저장
model.save("my_vector_model.keras")
