import streamlit as st
from PIL import Image
import io
from streamlit.components.v1 import html
from utils.db import write_sample, write_samples
from utils.models import load_model
import numpy as np
from utils.face_utils import get_face_coords, get_cossim, smart_resize, get_emb

model = load_model(chpt='app\checkpoints\infoVAE_39.pth')


def nav_page(page_name, timeout_secs=3):
    link = r'http://localhost:8501/90%9f%d0%be%d0%b8%d1%81%d0%ba'
    nav_script = r"""
        <script type="text/javascript">
            function attempt_nav_page(link){
                var link = "http://localhost:8501/90%9f%d0%be%d0%b8%d1%81%d0%ba";

                var links = window.parent.document.getElementsByTagName("a");

                for (var i = 0; i < links.length; i++) {
                    console.log(links[i].href.toLowerCase(), link);
                    if (links[i].href == link) {
                        links[i].click();
                        return;
                    }
                }

            }
            window.addEventListener("load", function() {
                attempt_nav_page("""+f'"{link}");'+"""
            });
        </script>
    """
    html(nav_script)


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

        x, y, w, h = get_face_coords(img)
        face = np.array(img)[y:y+h, x:x+w]
        face = smart_resize(face, 32)
        emb = list(get_emb(face, model).astype(float))
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
