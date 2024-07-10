## Реально или Фейк
Приложение для классификации изображений на "фальшивые" и "настоящие" с помощью предварительно обученной модели TensorFlow. Приложение использует библиотеку Tkinter для создания графического интерфейса и OpenCV для обработки изображений.

## Функции
Введите URL-адрес изображения для классификации в удобном GUI интерфейсе
Нажмите кнопку "Проверить", чтобы классифицировать изображение
Результат классификации будет отображен под кнопкой
Щелкните правой кнопкой мыши на поле ввода URL, чтобы вставить URL из буфера обмена или используйте комбинацию Ctrl+V

## Как использовать
Запустите скрипт main.py для запуска приложения.
Введите URL-адрес изображения в поле ввода URL-адреса.
Нажмите кнопку "Проверить", чтобы классифицировать изображение.
Результат классификации будет отображен под кнопкой.
## Примечание
Предварительно обученная модель, используемая в этом приложении, включена в данный репозиторий, но вы можете обучить свою собственную модель и сохранить ее под именем "model.keras" в каталоге "model".
Это приложение предполагает, что изображение по указанному URL-адресу находится в открытом доступе и может быть загружено с помощью библиотеки requests.
Обработка ошибок минимальна, и ошибки могут возникнуть, если изображение не может быть загружено или если модель не обучена должным образом.

Тестовая точность: 90.94%
