import os
import time
import uuid
import cv2

import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt

import albumentations as alb

# IMAGES_PATH = os.path.join('data', 'images')
# number_images = 100

# video_capture = cv2.VideoCapture(0)

# for image_number  in range (number_images):
#     print('Collecting image {}'.format(image_number))
#     ret, frame = video_capture.read()
#     image_name = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
#     cv2.imwrite(image_name, frame)
#     cv2.imshow('frame', frame)
#     time.sleep(0.5)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video_capture.release()
# cv2.destroyAllWindows()

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# tf.config.experimental.list_physical_devices("GPU")
# print(tf.test.is_gpu_available())

# images = tf.data.Dataset.list_files('data/images/*.jpg')

# # images.as_numpy_iterator().next()

# def load_image(x): 
#     byte_img = tf.io.read_file(x)
#     img = tf.io.decode_jpeg(byte_img)
#     return img

# images = images.map(load_image)

# print(images.as_numpy_iterator().next())

# print(type(images))

# image_generator = images.batch(10).as_numpy_iterator()

# plot_images = image_generator.next()

# fig, ax = plt.subplots(ncols=10, figsize=(20,20))
# for idx, image in enumerate(plot_images):
#     ax[idx].imshow(image) 
# plt.show()


# for folder in ['train','test','val']:
#     for file in os.listdir(os.path.join('data', folder, 'images')):
#         filename = file.split('.')[0]+'.json'
#         existing_filepath = os.path.join('data','labels', filename)
#         if os.path.exists(existing_filepath): 
#             new_filepath = os.path.join('data',folder,'labels',filename)
#             os.replace(existing_filepath, new_filepath) 


# augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
#                          alb.HorizontalFlip(p=0.5), 
#                          alb.RandomBrightnessContrast(p=0.2),
#                          alb.RandomGamma(p=0.2), 
#                          alb.RGBShift(p=0.2), 
#                          alb.VerticalFlip(p=0.5)], 
#                        bbox_params=alb.BboxParams(format='albumentations', 
#                                                   label_fields=['class_labels']))



cap = cv2.VideoCapture(1)
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500,:]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('EyeTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()