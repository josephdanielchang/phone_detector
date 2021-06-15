# Phone Detector - Train

# Command: python train_phone_finder.py ~/find_phone
# Input: 100 .jpg images (490x326) with labels.txt
# Output: trained model.h5

from find_phone import predict_phone_position
import sys
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

K.set_image_data_format('channels_first')  # set image order: numbers, channels, height, width

half_crop_size = 22
crop_size = 44

def crop_images(img, pos, num_sample_phones=50, num_sampe_background=50):

    height, width = img.shape
    phone_images = []
    background_images = []
    pos_pixel = np.array((int(pos[0] * width), int(pos[1] * height)))
    # left boundary of box
    box_lb = pos_pixel[0] - half_crop_size
    # right boundary of box
    box_rb = pos_pixel[0] + half_crop_size
    # upper boundary of box
    box_ub = pos_pixel[1] - half_crop_size
    # bottom boundary of box
    box_bb = pos_pixel[1] + half_crop_size
    # crop the phone from the image
    phone_crop = img[box_ub:box_bb, box_lb:box_rb]
    # randomly rotate 90 degree of cropped phone
    for i in range(num_sample_phones):
        random.seed(i)
        pi = random.random()
        if pi > 0.75:
            t = random.choice([1, 2, 3, 4])
            phone_images.append(np.rot90(phone_crop, t))
        else:
            phone_images.append(phone_crop)

    # randomly crop background images
    for i in range(num_sampe_background):
        # coordinate of the left up corner of cropped background
        random.seed(i)
        start_x = box_lb - 60 if (box_lb > 60) else 0
        start_y = box_ub - 60 if (box_ub > 60) else 0
        b_x = random.randint(start_x, width - crop_size)
        b_y = random.randint(start_y, height - crop_size)
        # in case there would be overlap between the background crop and phone crop
        while b_x in range(start_x, box_rb) and b_y in range(start_y, box_bb):
            b_x = random.randint(0, width - crop_size)
            b_y = random.randint(0, height - crop_size)
        back_crop = img[b_y: b_y + crop_size, b_x: b_x + crop_size]
        background_images.append(back_crop)

    return phone_images, background_images


def prepare_data(image_dir, label_dir):
    # read labels and store in list
    f = open(label_dir)
    iter_f = iter(f)
    list_f = []
    for line in iter_f:
        line = line.strip('\n')
        list_f.append(line.split(" "))
    # convert list to dict
    dict_f = {x[0]: np.array([round(float(x[1]), 4), round(float(x[2]), 4)]) for x in list_f}

    # read phone image data
    data_phone = []
    data_background = []
    for filename in os.listdir(image_dir):
        if filename != "labels.txt":
            image = cv2.imread(image_dir + '/' + filename)
            image_G = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            phone_images, background_images = crop_images(image_G, dict_f[filename])
            data_phone.extend(phone_images)
            data_background.extend(background_images)
    data_phone = np.array(data_phone)
    data_background = np.array(data_background)
    data = np.vstack((data_phone, data_background))
    label = np.hstack((np.ones(len(data_phone)), np.zeros(len(data_background))))

    data, label = shuffle(data, label, random_state=42)

    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=42)

    # reshape data to match input format of CNN
    train_data = train_data.reshape(train_data.shape[0], 1, crop_size, crop_size).astype('float32')
    test_data = test_data.reshape(test_data.shape[0], 1, crop_size, crop_size).astype('float32')
    # normalize input data
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    return train_data, test_data, train_label, test_label


def create_model(x_train, x_test, y_train, y_test):
    # for reproducible results
    np.random.seed(0)
    tf.random.set_seed(0)

    print('Train shape:', x_train.shape)
    print('Test shape:', x_test.shape)

    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1, 44, 44)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    # compile model
    sgd = optimizers.SGD(lr=0.1, decay=1e-2)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # earlystopping
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
    callbacks_list = [earlystop]
    model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=callbacks_list, epochs=50, batch_size=128)

    # save CNN model
    model.save("model_adam.h5")
    return model


def accuracy(image_dir, label_dir):
    # read labels and store in list
    f = open(label_dir)
    iter_f = iter(f)
    list_f = []
    for line in iter_f:
        line = line.strip('\n')
        list_f.append(line.split(" "))
    # convert list to dict
    dict_f = {x[0]: np.array([round(float(x[1]), 4), round(float(x[2]), 4)]) for x in list_f}

    # load CNN model
    model = load_model('model_adam.h5')

    # predict phone positions, update total accuracy
    accuracy = 0
    total = 0
    for filename in os.listdir(image_dir):
        total = total + 1
        print('file:', total)

        image = image_dir + '/' + filename
        pos = predict_phone_position(image, model)
        error = np.sqrt(np.sum(np.power(pos - dict_f[filename], 2)))
        if error <= 0.05:
            accuracy = accuracy + 1
        else:
            print(filename, " ", pos, " ", dict_f[filename])
        print('total correct so far:', accuracy)
    accuracy = accuracy / total
    print(accuracy)
    return accuracy


def main():
    # read path argument
    path = sys.argv[1]
    path = os.path.expanduser(path)
    print('*****************',path)

    # train model
    train_data, test_data, train_label, test_label = prepare_data(path, os.path.join(path, 'labels.txt'))
    create_model(train_data, test_data, train_label, test_label)

    # calculate test accuracy
    # accuracy(path, os.path.join(path, 'labels.txt'))


if __name__ == '__main__':
    main()

