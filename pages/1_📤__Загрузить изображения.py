import streamlit as st
from PIL import Image
import io
from streamlit.components.v1 import html
from utils.db import write_sample, write_samples
from utils.models import load_model
import numpy as np
from utils.face_utils import get_face_coords, get_cossim, smart_resize, get_emb, get_embedding

# model = load_model(chpt='checkpoints/infoVAE_39.pth')

st.title('Загрузить изображения лиц для поиска 👇')

uploaded_files = st.file_uploader(
    "Выберите файлы изображений", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

i = 0
if len(uploaded_files) > 0:
    cols = st.columns((len(uploaded_files)))
    embs, keys = [], []

    for uploaded_file, col in zip(uploaded_files, cols):
        bytes_data = uploaded_file.read()
        img = Image.open(io.BytesIO(bytes_data))

        # x, y, w, h = get_face_coords(img)
        # face = np.array(img)[y:y+h, x:x+w]
        # face = smart_resize(face, 32)
        emb = list(get_embedding(img).detach().numpy())
        embs.append(emb)
        keys.append(uploaded_file.name)

        with col:
            st.image(img, width=300)

        i += 1

    submit = st.button('Загрузить')
    write_samples(keys, embs)

    if submit:
        st.success(
            f'Признаки загруженных изображений сохранены в базу, теперь вы можете искать по ним. Для этого перейдите на страницу [поиска](/Поиск).', icon="✅")
