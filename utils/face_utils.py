import cv2
import numpy as np
from PIL import Image
import torch
import scipy.spatial as sp
from facenet_pytorch import MTCNN, InceptionResnetV1

model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160)


faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def smart_resize(input_image, new_size):
    input_image = Image.fromarray(input_image)
    width = input_image.width
    height = input_image.height

# Image is portrait or square
    if height >= width:
        crop_box = (0, (height-width)//2, width, (height-width)//2 + width)
        return input_image.resize(size=(new_size, new_size),
                                  box=crop_box)

# Image is landscape
    if width > height:
        crop_box = ((width-height)//2, 0, (width-height)//2 + height, height)

        return process_img(input_image.resize(size=(new_size, new_size),
                                              box=crop_box))


def process_img(img):
    img = torch.FloatTensor(np.array(img))
    img = torch.unsqueeze(img, dim=0)
    img = img.permute(0, 3, 1, 2)
    return img


def get_face_coords(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    x, y, w, h = faces[0]
    return x, y, w, h


def get_face(img, size=32):
    x, y, w, h = get_face_coords(img)
    face = img[y:y+h, x:x+w]
    h, w, c = face.shape
    face = np.array(smart_resize(face, size))
    return face


def get_emb(face, model):
    emb = model.encoder(process_img(face)).detach().numpy()
    emb = np.squeeze(emb, axis=0).reshape(4*512)
    return emb


def get_cossim(face1, face2, model):
    a = model.encoder(process_img(face1)).detach().numpy()
    b = model.encoder(process_img(face2)).detach().numpy()
    a = np.squeeze(a, axis=0).reshape(4*512)
    b = np.squeeze(b, axis=0).reshape(4*512)
    cos_sim = sp.distance.cosine(a, b)
    return cos_sim


def get_embedding(img):
    img_cropped = mtcnn(img)
    img_embedding = model(img_cropped.unsqueeze(0))
    return img_embedding


def get_sim(emb1, emb2):
    return 1-sp.distance.cosine(emb1.squeeze(0).detach().numpy(), emb2.squeeze(0).detach().numpy())
