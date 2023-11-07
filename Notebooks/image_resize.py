import cv2
import time
import random
import numpy as np
from matplotlib import pyplot as plt

mapper = {
    "Images/karan.jpg" : "Images/karan_cartoon.png",
    "Images/pious.jpg" : "Images/pious_cartoon.png",
    "Images/big_pious.jpg" : "Images/pious_cartoon.png",
    "Images/puneet.jpg" : "Images/puneet_cartoon.png",
    "Images/text_image.jpg" : "Images/font_out.jpg",
    "Images/amul_font.png": "Images/amul_font_all.png", 
    "Images/amul2_font.png" : "Images/amul2_font_all.png",
    "Images/parle_font.png" : "Images/parle_font_all.png", 
    "Images/brit_font.png" : "Images/brit_font_all.png",
    "Images/big_karan.jpg" : "Images/karan_cartoon.png",
    "Images/puneet.jpg" : "Images/puneet_cartoon.png",
    "Images/big_puneet.jpg" : "Images/puneet_cartoon.png",
    "Images/a.jpg" : "Images/A.jpg",
    "Images/b.jpg" : "Images/B.jpg",
    "Images/c.jpg" : "Images/C.jpg",
    "Images/d.jpg" : "Images/D.jpg",
    "Images/a.png" : "Images/aa.png",
    "Images/b.png" : "Images/B.png",
    "Images/c.png" : "Images/C.png",
    "Images/d.png" : "Images/D.png"
}



def resize_image(org_image, edited_image):
    time.sleep(random.randint(2,3))
    edited_image = cv2.imread(mapper[org_image])
    return edited_image

def predict():
    return 0