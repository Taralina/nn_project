import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
import time
import pandas as pd
import matplotlib.pyplot as plt

# Функция загрузки модели
def load_model(model_name, num_classes, model_path):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    
    # Принудительная загрузка модели на CPU, даже если она была обучена на GPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()  # Переводим модель в режим инференса
    return model

# Трансформации для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Функции предсказания
def predict_image(model, image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Функция загрузки изображения по URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Не удалось загрузить изображение: {e}")
        return None

# Функция для отображения логов обучения
def load_and_visualize_log(log_file):
    df = pd.read_csv(log_file)
    st.write("Логи обучения:")
    st.dataframe(df)

    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    ax[0].plot(df['Epoch'], df['Train_Loss'], label='Train Loss')
    ax[0].plot(df['Epoch'], df['Val_Loss'], label='Validation Loss')
    ax[0].set_title('Кривая потерь')
    ax[0].set_xlabel('Эпоха')
    ax[0].set_ylabel('Потери')
    ax[0].legend()

    ax[1].plot(df['Epoch'], df['Train_Accuracy'], label='Train Accuracy')
    ax[1].plot(df['Epoch'], df['Val_Accuracy'], label='Validation Accuracy')
    ax[1].set_title('Кривая точности')
    ax[1].set_xlabel('Эпоха')
    ax[1].set_ylabel('Точность')
    ax[1].legend()

    st.pyplot(fig)

# Загрузка моделей
model_recognition = load_model('resnet18', 11, 'model_resnet18.pth')
model_birds = load_model('densenet121', 200, 'model_desnet121.pth')

# Загружаем список классов
with open('birds_name.txt', 'r') as file:
    class_names_bird = eval(file.read())

class_names_recognition = [
    "роса", "туман, смог", "иней", "глазурь", "град", "молния", "дождь", "радуга", "изморозь", "песчаная буря", "снег"
]

# Главная функция для Streamlit UI
def main():
    page = st.sidebar.radio("Выберите страницу:", ("Природное явление", "Птички", "Информация"))

    # Контент главной страницы
    if page == "Природное явление":
        handle_weather_page()
    elif page == "Птички":
        handle_bird_page()
    elif page == "Информация":
        handle_info_page()

# Обработчик для страницы "Природное явление"
def handle_weather_page():
    st.header("Определение природного явления по картинке")
    uploaded_files, image = handle_image_upload()
    
    if uploaded_files or image:
        if st.button("Предсказать"):
            make_prediction(model_recognition, class_names_recognition, uploaded_files, image)

# Обработчик для страницы "Птички"
def handle_bird_page():
    st.header("Определение вида птички по картинке")
    uploaded_files, image = handle_image_upload()

    if uploaded_files or image:
        if st.button("Предсказать"):
            make_prediction(model_birds, class_names_bird, uploaded_files, image)

# Функция для загрузки изображения
def handle_image_upload():
    image_option = st.radio("Выберите способ загрузки изображения:", ("Загрузить изображение из файла", "Загрузить изображение по URL"))
    image = None
    uploaded_files = None

    if image_option == "Загрузить изображение из файла":
        uploaded_files = st.file_uploader("Выберите изображение...", type="jpg", accept_multiple_files=True)
        if uploaded_files:
            images = [Image.open(uploaded_file) for uploaded_file in uploaded_files]
            for i, img in enumerate(images):
                st.image(img, caption=f"Загруженное изображение {i+1}", use_column_width=True)

    elif image_option == "Загрузить изображение по URL":
        url = st.text_input("Введите URL изображения:")
        if url:
            image = load_image_from_url(url)
            if image:
                st.image(image, caption="Изображение из URL", use_column_width=True)
    
    return uploaded_files, image

# Функция для предсказания и отображения результата
def make_prediction(model, class_names, uploaded_files, image):
    start_time = time.time()
    if uploaded_files:
        for i, img in enumerate(uploaded_files):
            prediction = predict_image(model, img)
            predicted_class = class_names[prediction]
            st.write(f"Предсказание {i+1}: {predicted_class}")
    elif image:
        prediction = predict_image(model, image)
        predicted_class = class_names[prediction]
        st.write(f"Предсказание: {predicted_class}")
    else:
        st.warning("Пожалуйста, загрузите изображение.")

    elapsed_time = time.time() - start_time
    st.write(f"Время ответа модели: {elapsed_time:.2f} секунд")

# Обработчик для страницы "Информация"
def handle_info_page():
    st.header("Сводная информация по модели 'природные условия'")
    load_and_visualize_log('log_file')
    st.header("Сводная информация по модели 'птички'")
    load_and_visualize_log('log_file_dasnet')

if __name__ == "__main__":
    main()
