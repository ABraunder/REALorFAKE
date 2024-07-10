import tkinter as tk
from tkinter import messagebox
import requests
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf

class GUI:
    def __init__(self, master):
        # Создаем главное окно GUI
        self.master = master
        self.master.title("Классификатор Изображений")
        self.master.geometry("400x200")

        # Создаем метку для ввода URL
        self.url_label = tk.Label(master, text="Введите URL:")
        self.url_label.pack()

        # Создаем поле ввода для URL
        self.url_entry = tk.Entry(master, width=40)
        self.url_entry.pack()

        # Создаем кнопку для классификации изображения
        self.classify_button = tk.Button(master, text="Проверить", command=self.classify_image)
        self.classify_button.pack()

        # Создаем метку для вывода результата
        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        # Создаем контекстное меню для поля ввода URL
        self.url_entry.bind("<Button-3>", self.show_context_menu)

    def show_context_menu(self, event):
        # Создаем контекстное меню
        menu = tk.Menu(self.master, tearoff=0)
        menu.add_command(label="Вставить", command=self.paste_url)
        menu.tk_popup(event.x_root, event.y_root)

    def paste_url(self):
        # Вставляем URL из буфера обмена
        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, self.master.clipboard_get())

    def classify_image(self):
        # Получаем URL из поля ввода
        url = self.url_entry.get()
        if not url:
            # Если URL не введен, выводим ошибку
            messagebox.showerror("Ошибка", "Пожалуйста, введите URL")
            return

        try:
            # Отправляем запрос на получение изображения по URL
            response = requests.get(url)
            # Преобразуем ответ в массив байтов
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            # Декодируем массив байтов в изображение
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # Преобразуем изображение в формат RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize изображения до размера 64x64
            img = cv2.resize(img, (64, 64))
            # Нормализуем изображение
            img = img / 255.0 

            # Добавляем измерение для входного тензора
            img = np.expand_dims(img, axis=0)

            # Загружаем модель классификации
            model = tf.keras.models.load_model('model/model.keras')
            # Классифицируем изображение
            prediction = model.predict(img)

            # Округляем предсказание до целого числа
            predicted_class = np.round(prediction).astype(int)

            # Выводим результат классификации
            if predicted_class == 0:
                self.result_label.config(text="FAKE")
            else:
                self.result_label.config(text="REAL")
            # Очищаем поле ввода URL
            self.url_entry.delete(0, tk.END)
        except Exception as e:
            # Выводим ошибку, если она возникла
            messagebox.showerror("Ошибка", str(e))

root = tk.Tk()
gui = GUI(root)
root.mainloop()