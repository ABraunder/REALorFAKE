import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class TrainModel:
    def __init__(self, fake_dir, real_dir):
        # Директории для фальшивых и реальных изображений
        self.fake_dir = fake_dir
        self.real_dir = real_dir
        self.images = None
        self.labels = None
        self.model = None

    def load_data(self):
        # Загрузка данных из директорий
        fake_images = []
        real_images = []

        for filename in os.listdir(self.fake_dir):
            if filename.endswith('.jpg'):
                img = cv2.imread(os.path.join(self.fake_dir, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))
                fake_images.append(img)

        for filename in os.listdir(self.real_dir):
            if filename.endswith('.jpg'):
                img = cv2.imread(os.path.join(self.real_dir, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))
                real_images.append(img)

        fake_images = np.array(fake_images)
        real_images = np.array(real_images)

        # Создание меток для фальшивых и реальных изображений
        fake_labels = np.zeros(len(fake_images))
        real_labels = np.ones(len(real_images))

        self.images = np.concatenate((fake_images, real_images), axis=0)
        self.labels = np.concatenate((fake_labels, real_labels), axis=0)

        # Перемешивание данных
        indices = np.random.permutation(len(self.images))
        self.images = self.images[indices]
        self.labels = self.labels[indices]

    def create_model(self):
        # Создание модели нейронной сети
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self):
        # Обучение модели
        self.model.fit(self.images, self.labels, epochs=50, batch_size=64, validation_split=0.2)

        test_loss, test_acc = self.model.evaluate(self.images, self.labels, verbose=2)
        print(f'\nТестовая точность: {test_acc*100:.2f}%')

    def save_model(self, save_dir):
        # Сохранение модели
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'model.keras')
        self.model.save(model_path)

trainer = TrainModel('training_fake', 'training_real')
trainer.load_data()
trainer.create_model()
trainer.train_model()
trainer.save_model('model')