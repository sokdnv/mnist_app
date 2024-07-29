import torch
from model import Model
import streamlit as st
from preprocessing import preprocess
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2


@st.cache_resource()
def load_model():
    model = Model()
    model.load_state_dict(torch.load('mnist_loc.pt', map_location=torch.device('cpu')))
    return model


model = load_model()


def predict(img):
    img = preprocess(img)
    with torch.no_grad():
        pred, box = model(img)
    return pred, box


st.title('Локализуем ваши цифорки')
st.caption('с Серёжей')
st.divider()

with st.sidebar:
    st.caption('Я маленькая моделька (100 кб) которая училась на цифрах небольшого размера')
    st.caption('Пример ввода')
    st.image('demo.png')



with st.form(key='drawing'):
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        background_image= None,
        update_streamlit= True,
        height=400,
        drawing_mode='freedraw',
        key="canvas",
    )
    submit = st.form_submit_button('Вычислить!')

if submit:
    img = canvas_result.image_data.astype('uint8')
    img = Image.fromarray(img).convert('L')

    logit, box = predict(img)
    number = logit.argmax().item()

    st.write(f"Я считаю, что это {number}")

    img_np = np.array(img)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    box = box.cpu().numpy()[0]
    box = (box * [img_cv2.shape[1], img_cv2.shape[0], img_cv2.shape[1], img_cv2.shape[0]]).astype(int)
    img_cv2 = cv2.rectangle(img_cv2, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)

    st.image(img_cv2, caption='Изображение с локализацией')



