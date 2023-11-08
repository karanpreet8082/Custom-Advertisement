
# imports

import cv2
import math
import dlib
import numpy
import random
import keras_ocr
from PIL import Image
from imutils import face_utils
from matplotlib import pyplot as plt
from resize_image import image_resizer
from face_cartoonizer import cartoonifier
from keras.models import model_from_json


# Define Functions

# ------------  SORT POINTS  -----------------------------------------------------------------
def sort_points(points):
    points = sorted(points, key=lambda x: x[1])
    points = sorted(points[:2], key=lambda x: x[0]) + \
             sorted(points[2:], key=lambda x: x[0], reverse=True)
    return points

# ------------  DRAW REGION  -----------------------------------------------------------------
def draw_region(image, points):
    output = image.copy()
    points = sort_points(points)
    npoint = len(points)
    for i in range(npoint):
        cv2.line(output, points[i], points[(i + 1) % npoint], (0, 0, 255), 1, cv2.LINE_AA)
    for i in range(npoint):
        cv2.circle(output, points[i], 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(output, points[i], 4, (0, 255, 0), -1, cv2.LINE_AA)
    return output


# ------------  EXTRACT Region Of Interest  -----------------------------------------------------------------
def extract_roi(image, points):
    sp = sort_points(points)
    return image[sp[0][1]:sp[2][1], sp[0][0]:sp[2][0]]


# ------------  FIND MIDPOINT  -----------------------------------------------------------------
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)



# ------------  INPAINT TEXT  -----------------------------------------------------------------
def inpaint_text(img, pipeline):
    
    # generate (word, box) tuples 
    bounding_boxes = pipeline.recognize([img])
    
    # define mask
    mask = numpy.zeros(img.shape[:2], dtype="uint8")
    for box in bounding_boxes[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
        
    img = cv2.inpaint(img, mask, 20, cv2.INPAINT_NS)
#     img = cv2.inpaint(img, mask, 10, cv2.INPAINT_TELEA)
    return img




# ------------  BINARIZE  -----------------------------------------------------------------
def binarize(image, points=None, thresh=128, maxval=255, thresh_type=0):
    
    image = image.copy()
    if not points is None and type(points) is list and len(points) > 2:
        points = sort_points(points)
        points = numpy.array(points, numpy.int64)
        mask = numpy.zeros_like(image, numpy.uint8)
        cv2.fillConvexPoly(mask, points, (255, 255, 255), cv2.LINE_AA)
        image = cv2.bitwise_and(image, mask)

    msers = cv2.MSER_create().detectRegions(image)[0]
    setyx = set()
    for region in msers:
        for point in region:
            setyx.add((point[1], point[0]))
    setyx = tuple(numpy.transpose(list(setyx)))
    mask1 = numpy.zeros(image.shape, numpy.uint8)
    mask1[setyx] = maxval

    mask2 = cv2.threshold(image, thresh, maxval, thresh_type)[1]
    image = cv2.bitwise_and(mask1, mask2)
    return image


# ------------  CARTOON CLASSIFIER  -----------------------------------------------------------------
def cartoon_predictor(img, facial_predictors):
    input_text = alphabet
    image_scaled = img.copy()
    image_edit = image_scaled.copy()
    image_gray = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2GRAY)
    image_mask = binarize(image_gray, [], thresh, 255, 0)
    
    region_f = img
    tensor_f = image2tensor(region_f, NET_CARTOON.input_shape[0][1:3], .1, 1.)
    onehot_f = char2onehot(input_text[0], alphabet)
#     output_f = NET_CARTOON.predict([tensor_f, onehot_f])
#     output_f = np.squeeze(output_f)
#     output_f = np.asarray(output_f, np.uint8)

    o_layout = numpy.zeros_like(image_mask, numpy.uint8)
    inpainted_image = Image.fromarray(o_layout)
    
    inpainted_image = image_resizer(xyz_path, inpainted_image)
    
    return inpainted_image


# ------------  FIND CONTOURS  -----------------------------------------------------------------
def find_contours(image, min_area=0, sort=True):
    image = image.copy()
 
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    
    if len(contours) < 1:
        return ([], [])
    if sort:
        bndboxes = [cv2.boundingRect(contour) for contour in contours]
        contours, bndboxes = zip(*sorted(zip(contours, bndboxes), key=lambda x: x[1][0]))
    
    return contours, bndboxes


# ------------  DRAW CONTOURS  -----------------------------------------------------------------
def draw_contours(image, contours, index, color=(0, 255, 0), color_mode=None):
    image = cv2.cvtColor(image, color_mode) if color_mode else image.copy()
    drawn = numpy.zeros_like(image, numpy.uint8)
    for i in range(len(contours)):
        drawn = cv2.drawContours(drawn, contours, i, (255, 255, 255), -1, cv2.LINE_AA)
    if len(contours) > 0 and index >= 0:
        drawn = cv2.drawContours(drawn, contours, index, color, -1, cv2.LINE_AA)
    image = cv2.bitwise_and(drawn, image)
    return image



# ------------  GRAB REGION  -----------------------------------------------------------------
def grab_region(image, bwmask, contours, bndboxes, index):
    region = numpy.zeros_like(bwmask, numpy.uint8)
    if len(contours) > 0 and len(bndboxes) > 0 and index >= 0:
        x, y, w, h = bndboxes[index]
        region = cv2.drawContours(region, contours, index, (255, 255, 255), -1, cv2.LINE_AA)
        region = region[y:y+h, x:x+w]
        bwmask = bwmask[y:y+h, x:x+w]
        bwmask = cv2.bitwise_and(region, region, mask=bwmask)
        region = image[y:y+h, x:x+w]
        region = cv2.bitwise_and(region, region, mask=bwmask)
    return region


# ------------  GRAB REGION(s)  -----------------------------------------------------------------
def grab_regions(image, image_mask, contours, bndboxes):
    regions = []
    for index in range(len(bndboxes)):
        regions.append(grab_region(image, image_mask, contours, bndboxes, index))
    return regions


# ------------  IMAGE -> TENSOR  -----------------------------------------------------------------
def image2tensor(image, shape, padding=0.0, rescale=1.0, color_mode=None):
    output = cv2.cvtColor(image, color_mode) if color_mode else image.copy()
    output = numpy.atleast_3d(output)
    rect_w = output.shape[1]
    rect_h = output.shape[0]
    sqrlen = int(numpy.ceil((1.0 + padding) * max(rect_w, rect_h)))
    sqrbox = numpy.zeros((sqrlen, sqrlen, output.shape[2]), numpy.uint8)
    rect_x = (sqrlen - rect_w) // 2
    rect_y = (sqrlen - rect_h) // 2
    sqrbox[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = output
    output = cv2.resize(sqrbox, shape[:2])
    output = numpy.atleast_3d(output)
    output = numpy.asarray(output, numpy.float32) * rescale
    output = output.reshape((1,) + output.shape)
    return output


# ------------  CHAR -> ONEHOT  -----------------------------------------------------------------
def char2onehot(character, alphabet):
    onehot = [0.] * len(alphabet)
    onehot[alphabet.index(character)] = 1.
    onehot = numpy.asarray(onehot, numpy.float32).reshape(1, len(alphabet), 1)
    return onehot



# ------------  TRANSFER COLOR  -----------------------------------------------------------------
def transfer_color_max(source, target):
    colors = source.convert('RGB').getcolors(256*256*256)
    colors = sorted(colors, key=lambda x: x[0], reverse=True)
    maxcol = colors[0][1] if len(colors) == 1 else \
             colors[0][1] if colors[0][1] != (0, 0, 0) else \
             colors[1][1]
    output = Image.new('RGB', target.size)
    colors = Image.new('RGB', target.size, maxcol)
    output.paste(colors, (0, 0), target.convert('L'))
    return output



# ------------  RESIZE  -----------------------------------------------------------------
def resize(image, w=-1, h=-1, bbox=False):
    image = Image.fromarray(image)
    bnbox = image.getbbox() if bbox else None
    image = image.crop(bnbox) if bnbox else image
    if w <= 0 and h <= 0:
        w = image.width
        h = image.height
    elif w <= 0 and h > 0:
        w = int(image.width / image.height * h)
    elif w > 0 and h <= 0:
        h = int(image.height / image.width * w)
    else:
        pass
    image = image.resize((w, h))
    image = numpy.asarray(image, numpy.uint8)
    return image



# ------------  PASTE PATCH ON IMAGE  -----------------------------------------------------------------
def paste_images(image, patches, bndboxes):
    image = Image.fromarray(image)
    for patch, bndbox in zip(patches, bndboxes):
        patch = Image.fromarray(patch)
        image.paste(patch, bndbox[:2])
    image = numpy.asarray(image, numpy.uint8)
    return image



# ------------  INPAINTING  -----------------------------------------------------------------
def inpaint(image, mask):
    k = numpy.ones((5, 5), numpy.uint8)
    m = cv2.dilate(mask, k, iterations=1)
    i = cv2.inpaint(image, m, 10, cv2.INPAINT_TELEA)
    return i



# ------------  TRANSFER COLOR PALET  -----------------------------------------------------------------
def transfer_color_pal(source, target):
    source = source.convert('RGB')
    src_bb = source.getbbox()
    src_bb = source.crop(src_bb) if src_bb else source.copy()
    colors = Image.new('RGB', src_bb.size)
    src_np = numpy.asarray(src_bb, numpy.uint8)
    for i in range(src_np.shape[0]):
        row_np = src_np[i].reshape(1, -1, 3)
        col_id = numpy.where(row_np == 0)[1]
        row_np = numpy.delete(row_np, col_id, axis=1)
        row_im = Image.fromarray(row_np).resize((colors.width, 1))
        colors.paste(row_im, (0, i))
    target = target.convert('L')
    colors = colors.resize(target.size)
    output = Image.new('RGB', target.size)
    output.paste(colors, (0, 0), target)
    return output



# ------------  TEXT CREATOR WITH FONT  -----------------------------------------------------------------
def text_creator_with_font(img, points, alphabet, input_text, img_path):
    image_scaled = img.copy()
    image_edit = image_scaled.copy()
    image_gray = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2GRAY)
    image_mask = binarize(image_gray, [], thresh, 255, 0)
    contours, bndboxes = find_contours(image_mask, cntmin)
    
    region_f = grab_region(image_mask, image_mask, contours, bndboxes, 0)
    tensor_f = image2tensor(region_f, NET_F.input_shape[0][1:3], .1, 1.)
    onehot_f = char2onehot(input_text[0], alphabet)
    output_f = NET_F.predict([tensor_f, onehot_f])
    output_f = numpy.squeeze(output_f)
    output_f = numpy.asarray(output_f, numpy.uint8)
    
    region_c = grab_region(img, image_mask, contours, bndboxes, index)
    source_c = Image.fromarray(region_c)
    target_f = Image.fromarray(output_f)
    output_c = transfer_color_max(source_c, target_f)
    output_c = numpy.asarray(output_c, numpy.uint8)
    
    output_f = resize(output_f, -1, region_f.shape[0], True)
    output_c = resize(output_c, -1, region_c.shape[0], True)
    
    # inpaint old layout
    mpatches = grab_regions(image_mask, image_mask, contours, bndboxes)
    o_layout = numpy.zeros_like(image_mask, numpy.uint8)
    o_layout = paste_images(o_layout, mpatches, bndboxes)
    inpainted_image = inpaint(img, o_layout)
    
    # generate final result
    inpainted_image = Image.fromarray(inpainted_image)
    
    inpainted_image = image_resizer(img_path, inpainted_image)
    
    return inpainted_image



# ------------  CREATE ALL ALPHABETS  -----------------------------------------------------------------
def alphabet_with_font(img_path, alphabet):
    img = cv2.imread(xyz_path)
    input_text = alphabet
    image_scaled = img.copy()
    image_edit = image_scaled.copy()
    image_gray = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2GRAY)
    image_mask = binarize(image_gray, [], thresh, 255, 0)
    contours, bndboxes = find_contours(image_mask, cntmin)
    
    region_f = grab_region(image_mask, image_mask, contours, bndboxes, 0)
    tensor_f = image2tensor(region_f, NET_F.input_shape[0][1:3], .1, 1.)
    onehot_f = char2onehot(input_text[0], alphabet)
    output_f = NET_F.predict([tensor_f, onehot_f])
    output_f = numpy.squeeze(output_f)
    output_f = numpy.asarray(output_f, numpy.uint8)
    
    region_c = grab_region(img, image_mask, contours, bndboxes, 0)
    source_c = Image.fromarray(region_c)
    target_f = Image.fromarray(output_f)
    output_c = transfer_color_max(source_c, target_f)
    output_c = numpy.asarray(output_c, numpy.uint8)
    
    output_f = resize(output_f, -1, region_f.shape[0], True)
    output_c = resize(output_c, -1, region_c.shape[0], True)

    mpatches = grab_regions(image_mask, image_mask, contours, bndboxes)
    o_layout = numpy.zeros_like(image_mask, numpy.uint8)
    o_layout = paste_images(o_layout, mpatches, bndboxes)
    inpainted_image = inpaint(img, o_layout)
    
    inpainted_image = Image.fromarray(inpainted_image)
    
    inpainted_image = image_resizer(img_path, inpainted_image)
    
    return inpainted_image


# ------------  LOAD MODELS  -----------------------------------------------------------------

with open('models/fannet.json', 'r') as fp:
    NET_F = model_from_json(fp.read())
with open('models/colornet.json', 'r') as fp:
    NET_C = model_from_json(fp.read())
with open('models/cartoon_model.json', 'r') as fp:
    NET_CARTOON = model_from_json(fp.read())
    
NET_F.load_weights('models/fannet_weights.h5')
NET_C.load_weights('models/colornet_weights.h5')
NET_CARTOON.load_weights('models/cartoon_weights.h5')

# ------  INITIALISE VARIABLES  -----------------------------------------------------



index = 0
factor = 55
cntmin = 25
thresh = 150
xyz_path = 'Images/text_image.jpg'
points = [(125,10),(1100, 10),(1100, 100),(125, 100)]
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
input_text = "SOME RANDOM TEXT HERE"
p = "shape_predictor_68_face_landmarks.dat"



if __name__ == "__main__":
    
    img_path = 'Images/text_image.jpg'
    # img_path = input("Path to original advertisement")
    
    # Font Generator
    # input_text = input("Enter the input text to write on Image : ")
    
    input_ad = cv2.imread(img_path)
    font_img = extract_roi(input_ad, points)
    new_text = text_creator_with_font(input_ad, points, alphabet, input_text, img_path)
    
    
    # Text removal
    pipeline = keras_ocr.pipeline.Pipeline()
    output = keras_ocr.tools.read(img_path)
    output_textless = inpaint_text(input_ad,pipeline)
    
    
    output_textless[ points[0][1]:points[2][1], points[0][0]:points[2][0] ] = new_text
    
    
    # Face extractor
    
    face_image_path = "Images/karan.jpg"
    # face_image_path = input("Enter the path to face image")
    xyz_path = face_image_path
    
    face = cv2.imread(face_image_path)
    
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale image
    # minSize : Face smaller than this will not be detected
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    bndboxes = faces.tolist()
    
    points = [(bndboxes[0][0],bndboxes[0][1]),
          (bndboxes[0][0]+bndboxes[0][2], bndboxes[0][1])
          ,(bndboxes[0][0]+bndboxes[0][2], bndboxes[0][1]+bndboxes[0][3]),
          (bndboxes[0][0], bndboxes[0][1]+bndboxes[0][3])]
    
    # Cropping the face

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        x, y = max(0, x-factor), max(0, y-factor)
        w += (2*factor)
        h += (2*factor)
        cropped_face = face[y:y+h, x:x+w]
    
        #cv2.imwrite("Images/face"+str(random.randint(0,10000))+".jpg", cropped_face)
        print("Face cropped and saved successfully.")
    else:
        print("No face detected in the input image.")
    
    
    # instance of a pre-trained frontal face detector
    # the facial landmark predictor


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    
    image = cropped_face.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region
        # return 68 values that are 68 face landmarks
        facial_features = predictor(gray, rect)
        facial_features = face_utils.shape_to_np(facial_features)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in facial_features:
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
            
    cartoon = cartoon_predictor(cropped_face, facial_features)
    
    indentifier = str(random.randint(0, 9999))
    
    
    # Save files to local
    cv2.imwrite("Output/cartoon"+ indentifier +".png", cartoon)
    cv2.imwrite("Output/out_ad"+ indentifier +".png", output_textless)
    cv2.imwrite("Output/cartoonbonus"+ indentifier +".png", cartoonifier(cropped_face))