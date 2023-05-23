import streamlit as st
from PIL import Image
import io
from utils.face_utils import get_face_coords, get_sim, smart_resize, get_embedding
import numpy as np

st.set_page_config(
    page_title="FaceCompare",
    page_icon="👁️",
)

st.write("# Добро пожаловать в FaceCompare! 🧑↔️👩")

st.info('Продолжая пользоваться данным приложением вы соглашаетесь на обработку персональных данных', icon="ℹ️")
# st.sidebar.success("Select a demo above.")


with st.expander("Подробнее"):
    st.markdown(
        """
    FaceCompare позволяет сравнивать изображения лиц людей и искать изображения с наибольшим сходством.

    Для поиска по фотографии изображений загрузите свои изображения на странице **Загрузить изображения**.
    """
    )


st.markdown('### Загрузите два изображения для сравнения')
uploaded_files = st.file_uploader(
    "Выберите файлы изображений", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

i = 0
faces = []
if len(uploaded_files) != 2:
    st.error('Пожалуйста, загрузите ровно 2 изображения!', icon="🚨")
else:
    cols = st.columns((len(uploaded_files)))
    embs = []

    for uploaded_file, col in zip(uploaded_files, cols):
        bytes_data = uploaded_file.read()
        img = Image.open(io.BytesIO(bytes_data))
        img_emb = get_embedding(img)
        embs.append(img_emb)

        with col:
            st.image(img, width=300)

        i += 1

    submit = st.button('Загрузить')
    # model = load_model(chpt='checkpoints/infoVAE_39.pth')
    # face1, face2 = faces
    # face1, face2 = smart_resize(face1, 32), smart_resize(face2, 32)

    if submit:
        sim = get_sim(embs[0], embs[1])
        st.success(f'Сходство: {sim:.3f}', icon="✅")

st.write("---")
st.write("""<div style="width:100%;text-align:center;"><a href="https://github.com/marianna13/IMDB_faces" style="float:center"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="22px"></img></a></div>""", unsafe_allow_html=True)
