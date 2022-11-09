import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory as image_data

model =tf.keras.models.load_model('playing_card.h5')

batch_size = 32
img_height = 180
img_width = 180

class_names = ['ace of clubs', 'ace of diamonds', 'ace of hearts', 'ace of spades', 'eight of clubs', 'eight of diamonds', 
                'eight of hearts', 'eight of spades', 'five of clubs', 'five of diamonds', 'five of hearts', 'five of spades',
                'four of clubs', 'four of diamonds', 'four of hearts', 'four of spades', 'jack of clubs', 'jack of diamonds',
                'jack of hearts', 'jack of spades', 'king of clubs', 'king of diamonds', 'king of hearts', 'king of spades', 
                'nine of clubs', 'nine of diamonds', 'nine of hearts', 'nine of spades', 'queen of clubs', 'queen of diamonds', 
                'queen of hearts', 'queen of spades', 'seven of clubs', 'seven of diamonds', 'seven of hearts', 'seven of spades', 
                'six of clubs', 'six of diamonds', 'six of hearts', 'six of spades', 'ten of clubs', 'ten of diamonds', 'ten of hearts', 
                'ten of spades', 'three of clubs', 'three of diamonds', 'three of hearts', 'three of spades', 'two of clubs', 
                'two of diamonds', 'two of hearts', 'two of spades']


def Card_Identifier(image_directory):
    '''
    the image has to be placed in a folder inside a folder
    e.g main/sub/picture.jpg
    the argument to be passed will be "main"
    '''
    image = image_data(
    image_directory,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    prediction = model.predict(image)

    for i in range(len(prediction)):
        print(class_names[np.argmax(prediction[i])])

Card_Identifier('your directory folder')