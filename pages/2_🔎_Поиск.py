import streamlit as st
from PIL import Image
import io
from utils.search import search
from utils.face_utils import get_face_coords, smart_resize
import numpy as np


st.title('Поиск 🔎')

st.markdown('Загрузить изображение для поиска в базе')

uploaded_file = st.file_uploader(
    "Выберите файл изображения", type=['png', 'jpg', 'jpeg'])

if uploaded_file:

    bytes_data = uploaded_file.read()
    img = Image.open(io.BytesIO(bytes_data))
    x, y, w, h = get_face_coords(img)
    face = np.array(img)[y:y+h, x:x+w]
    face = smart_resize(face, 32)
    st.image(np.array(img)[y:y+h, x:x+w], width=300)

    submit = st.button('Поиск')
    if submit:
        st.markdown('### Результаты поиска:')
        results = search(uploaded_file.name, face)
        st.write(results)
