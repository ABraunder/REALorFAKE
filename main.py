import tkinter as tk
from tkinter import messagebox
import requests
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf

class GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Классификатор Изображений")
        self.master.geometry("400x300")

        self.url_label = tk.Label(master, text="Введите URL:")
        self.url_label.pack()

        self.url_entry = tk.Entry(master, width=40)
        self.url_entry.pack()
        self.url_entry.bind("<Control-V>", self.paste_url)
        self.url_entry.bind("<Button-3>", self.show_context_menu)

        self.classify_button = tk.Button(master, text="Проверить", command=self.classify_image)
        self.classify_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.correct_label = tk.Label(master, text="Исправить:")
        self.correct_label.pack()

        self.correct_button = tk.Button(master, text="Предложить исправление", command=self.show_correction_buttons, state="disabled")
        self.correct_button.pack()

        self.fake_button = tk.Button(master, text="Fake", command=lambda: self.submit_correction("Fake"))
        self.fake_button.pack_forget()

        self.real_button = tk.Button(master, text="Real", command=lambda: self.submit_correction("Real"))
        self.real_button.pack_forget()

        self.submit_button = tk.Button(master, text="Отправить исправление", command=self.submit_correction, state="disabled")
        self.submit_button.pack_forget()

        self.url_entry.bind("<Button-3>", self.show_context_menu)

    def show_context_menu(self, event):
        menu = tk.Menu(self.master, tearoff=0)
        menu.add_command(label="Вставить", command=self.paste_url)
        menu.tk_popup(event.x_root, event.y_root)

    def paste_url(self):
        if self.master.clipboard_get():
            self.url_entry.delete(0, tk.END)
            self.url_entry.insert(0, self.master.clipboard_get())
        else:
            messagebox.showerror("Ошибка", "Пустой буфер обмена или нет возможности вставить URL")

    def classify_image(self):
        url = self.url_entry.get()
        if not url:
            messagebox.showerror("Ошибка", "Пожалуйста, введите URL")
            return

        try:
            response = requests.get(url)
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            self.img = np.expand_dims(img, axis=0)

            self.model = tf.keras.models.load_model('model/model.keras')
            prediction = self.model.predict(self.img)
            predicted_class = np.round(prediction).astype(int)

            if predicted_class == 0:
                self.result_label.config(text="FAKE")
            else:
                self.result_label.config(text="REAL")

            self.url_entry.delete(0, tk.END)
            self.classify_button.config(state="normal")
            self.correct_button.config(state="normal")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def show_correction_buttons(self):
        self.correct_button.pack_forget()
        button_frame = tk.Frame(self.master)
        button_frame.pack()
        self.fake_button.pack(side=tk.LEFT, padx=10)
        self.real_button.pack(side=tk.RIGHT, padx=10)
        self.submit_button.pack()

    def submit_correction(self, correct_class=None):
        predicted_class = self.result_label.cget("text")
        if correct_class:
            self.update_model(predicted_class, correct_class)
        self.fake_button.pack_forget()
        self.real_button.pack_forget()
        self.submit_button.pack_forget()
        self.correct_button.pack()
        self.classify_button.config(state="normal")
        self.correct_button.config(state="disabled")

    def update_model(self, predicted_class, correct_class):
        img = self.img
        label = np.array([1 if correct_class == "REAL" else 0])
        self.model.fit(img, label, epochs=1, verbose=0)
        self.model.save('model/model.keras')
        print("Модель обновлена!")

root = tk.Tk()
gui = GUI(root)
root.mainloop()