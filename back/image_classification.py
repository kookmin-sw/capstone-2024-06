import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16

data_dir = "back/image"
batch_size = 32
img_height = 224
img_width = 224

train_data_gen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    validation_split=0.2)  # 20%는 검증 데이터로 사용

train_generator = train_data_gen.flow_from_directory(data_dir,
                                                     target_size=(img_height, img_width),
                                                     batch_size=batch_size,
                                                     class_mode='categorical',
                                                     subset='training')

val_generator = train_data_gen.flow_from_directory(data_dir,
                                                   target_size=(img_height, img_width),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   subset='validation')

num_classes = len(train_generator.class_indices)

weights_path = "back/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

# VGG16 모델 로드
base_model = VGG16(weights=weights_path, include_top=False, input_shape=(img_height, img_width, 3))

# VGG16 모델의 특징 추출 부분을 고정
for layer in base_model.layers:
    layer.trainable = False

# 새로운 모델 생성 (VGG16 기반)
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # 드롭아웃 추가
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 30
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# 모델 저장
model.save("my_model")