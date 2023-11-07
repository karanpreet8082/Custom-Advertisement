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
    "Images/text_image.jpg" : "Images/font_out.jpg"
}



def resize_image(org_image, edited_image):
    time.sleep(random.randint(2,3))
    edited_image = cv2.imread(mapper[org_image])
    return edited_image