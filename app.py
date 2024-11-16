import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import torch.nn as nn
import time

# Загрузка модели для природного явления
model_recognition = models.resnet18(pretrained=False)
model_recognition.fc = torch.nn.Linear(model_recognition.fc.in_features, 11) 
model_recognition.load_state_dict(torch.load('model_resnet18.pth', map_location='cpu'))
model_recognition.eval()

# Загрузка модели для определения птички
model_birds = models.densenet121(pretrained=False)
model_birds.classifier = nn.Linear(model_birds.classifier.in_features, 200)  # 200 классов (CUB-200-2011)
checkpoint = torch.load('model_desnet121.pth', map_location='cpu')
model_birds.load_state_dict(checkpoint, strict=False)  # Игнорируем несоответствия в весах

model_birds.eval()


# Трансформации для входных изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Используем те же параметры нормализации
])


# Функция предсказания
def predict(image):
    image = transform(image).unsqueeze(0)  # Добавляем batch-освобождение
    with torch.no_grad():
        output = model_recognition(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Функция предсказания для птицы
def predict_birds(image):
    image = transform(image).unsqueeze(0)  # Добавляем batch-освобождение
    with torch.no_grad():
        output = model_birds(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()


# Функция для загрузки изображения по URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

class_names_recognition = ["роса", "туман, смог", "иней", "глазурь", "град", "молния", "дождь",
"радуга", "изморозь", "песчаная буря", "снег"]

# Чтение данных из текстового файла
with open('birds_name.txt', 'r') as file:
    # Прочитаем строку
    data = file.read()
class_names_bird = eval(data)

#визуализаци для погоды
def load_and_visualize_log(log_file):
    # Загрузка данных из CSV
    df = pd.read_csv(log_file)
    
    # Отображение таблицы с метриками
    st.write("Логи обучения:")
    st.dataframe(df)
    
    # Построение графиков кривых потерь и точности
    plt.style.use("seaborn-v0_8-talk")
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    # График потерь
    ax[0].plot(df['Epoch'], df['Train_Loss'], label='Train_Loss')
    ax[0].plot(df['Epoch'], df['Val_Loss'], label='Validation_Loss')
    ax[0].set_title('Кривая потерь')
    ax[0].set_xlabel('Эпоха')
    ax[0].set_ylabel('Потери')
    ax[0].legend()

    # График точности
    ax[1].plot(df['Epoch'], df['Train_Accuracy'], label='Train_Accuracy')
    ax[1].plot(df['Epoch'], df['Val_Accuracy'], label='Validation_Accuracy')
    ax[1].set_title('Кривая точности')
    ax[1].set_xlabel('Эпоха')
    ax[1].set_ylabel('Точность')
    ax[1].legend()

    st.pyplot(fig)

    # F1-метрика
    fig_f1, ax_f1 = plt.subplots(figsize=(10, 6))
    ax_f1.plot(df['Epoch'], df['train_f1'], label='Train F1')
    ax_f1.plot(df['Epoch'], df['val_f1'], label='Val F1')
    ax_f1.set_title('Кривая F1-метрики')
    ax_f1.set_xlabel('Эпоха')
    ax_f1.set_ylabel('F1 Score')
    ax_f1.legend()

    st.pyplot(fig_f1)

uploaded_files = None
image = None 
# Создание боковой панели для навигации
page = st.sidebar.radio("Выберите страницу:", ("Природное явление", "Птички", "Информация"))

# Контент главной страницы
if page == "Природное явление":
    st.header("Определение природного явления по картинке")
    # Выбор способа загрузки
    image_option = st.radio(
    "Выберите способ загрузки изображения:",
    ("Загрузить изображение из файла", "Загрузить изображение по URL")
    )   
    if image_option == "Загрузить изображение из файла":
    # Загрузка изображения из файла
        uploaded_files = st.file_uploader("Выберите изображение...", type="jpg", accept_multiple_files=True)
        if uploaded_files:
            images = [Image.open(uploaded_file) for uploaded_file in uploaded_files]
            for i, img in enumerate(images):
                st.image(img, caption=f"Загруженное изображение {i+1}", use_column_width=True)

    elif image_option == "Загрузить изображение по URL":
    # Загрузка изображения по URL
        url = st.text_input("Введите URL изображения:")
        if url:
            try:
                image = load_image_from_url(url)
                st.image(image, caption="Изображение из URL", use_column_width=True)
            except Exception as e:
                st.error(f"Не удалось загрузить изображение. Ошибка: {e}")

# Предсказание
    if st.button("Предсказать"):
        if uploaded_files:  # Если загружены изображения из файла
            start_time = time.time()  # Засекаем время начала предсказания
            for i, img in enumerate(images):
                prediction = predict(img)
                predicted_class = class_names_recognition[prediction]  # Получаем название класса
                st.write(f"Природное явление {i+1}: {predicted_class}")
            end_time = time.time()  # Засекаем время окончания предсказания
            elapsed_time = end_time - start_time
            st.write(f"Время ответа модели: {elapsed_time:.2f} секунд")
        elif url:  # Если введен URL изображения
            start_time = time.time()  # Засекаем время начала предсказания
            prediction = predict(image)
            predicted_class = class_names_recognition[prediction]  # Получаем название класса
            st.write(f"Предсказание: {predicted_class}")
            end_time = time.time()  # Засекаем время окончания предсказания
            elapsed_time = end_time - start_time
            st.write(f"Время ответа модели: {elapsed_time:.2f} секунд")
        else:
            st.warning("Пожалуйста, загрузите изображение.")

# Контент второй страницы
elif page == "Птички":
    st.header("Определение вида птички по картинке")
    # Выбор способа загрузки
    image_option = st.radio(
    "Выберите способ загрузки изображения:",
    ("Загрузить изображение из файла", "Загрузить изображение по URL")
    )   
    if image_option == "Загрузить изображение из файла":
    # Загрузка изображения из файла
        uploaded_files = st.file_uploader("Выберите изображение...", type="jpg", accept_multiple_files=True)
        if uploaded_files:
            images = [Image.open(uploaded_file) for uploaded_file in uploaded_files]
            for i, img in enumerate(images):
                st.image(img, caption=f"Загруженное изображение {i+1}", use_column_width=True)

    elif image_option == "Загрузить изображение по URL":
        # Загрузка изображения по URL
        url = st.text_input("Введите URL изображения:")
        if url:
            try:
                # Попробуем загрузить изображение по URL
                response = requests.get(url)
                response.raise_for_status()  # Проверка на успешный запрос (status code 200)

                # Проверим, является ли контент изображением
                if 'image' not in response.headers['Content-Type']:
                    raise ValueError("Это не изображение.")
                
                img = Image.open(BytesIO(response.content))  # Открываем изображение
                image = img
                st.image(img, caption="Изображение из URL", use_column_width=True)
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка при загрузке изображения: {e}")
            except Exception as e:
                st.error(f"Не удалось обработать изображение: {e}")
              
    # Предсказание
    if st.button("Предсказать"):
        if uploaded_files:  # Если загружены изображения из файла
            start_time = time.time()  # Засекаем время начала предсказания
            for i, img in enumerate(images):
                prediction = predict_birds(img)
                predicted_class = class_names_bird[prediction]  # Получаем название класса
                st.write(f"Ваша птичка {i+1}: {predicted_class}")
            end_time = time.time()  # Засекаем время окончания предсказания
            elapsed_time = end_time - start_time
            st.write(f"Время ответа модели: {elapsed_time:.2f} секунд")
        elif image:  # Если введен URL изображения и оно успешно загружено
            start_time = time.time()  # Засекаем время начала предсказания
            prediction = predict_birds(image)
            predicted_class = class_names_bird[prediction]  # Получаем название класса
            st.write(f"Предсказание: {predicted_class}")
            end_time = time.time()  # Засекаем время окончания предсказания
            elapsed_time = end_time - start_time
            st.write(f"Время ответа модели: {elapsed_time:.2f} секунд")
        else:
            st.warning("Пожалуйста, загрузите изображение.")

# Контент третьей страницы
elif page == "Информация":
    #Загрузка логов по погоде
    log_file = 'log_file'
    st.header("Сводная информация по модели 'природные условия'")
    st.write("Время обучения нейронной сети: 87.910699 секунд")
    load_and_visualize_log(log_file)

    st.header("Сводная информация по модели 'птички'")
    st.write("Время обучения нейронной сети: 488.213757 секунд секунд")
    log_file_d = 'log_file_dasnet'
    load_and_visualize_log(log_file_d)

