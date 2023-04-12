import json
import glob
from utils.face_utils import get_emb, get_cossim, get_face_coords, smart_resize
from utils.models import load_model
import numpy as np
import scipy.spatial as sp
from utils.db import query_sample

model = load_model(chpt='checkpoints/infoVAE_39.pth')


def get_data():
    data = []
    for js in glob.glob(r'C:\Users\HP\Documents\NN\convnet\app\DB\*.json'):
        with open(js) as f:
            d = json.load(f)
            data.append(d)
    return data


def search(img_name, img_to_search):

    emb = get_emb(img_to_search, model)
    matches = query_sample(img_name, emb)
    return matches
