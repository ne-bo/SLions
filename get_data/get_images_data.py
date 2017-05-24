import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.misc import pilutil
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle
import tensorflow as tf
import skimage.feature
import cv2
import datetime

path_to_train_small = '/home/natasha/SeaLions/TrainSmall2/Train'
path_to_train_dot_small = '/home/natasha/SeaLions/TrainSmall2/TrainDotted/'
# width = 5616
# height = 3744
width = 3328
height = 4992
pixel_depth = 255.0
window_size = 104
number_of_channels = 3
color_delta = 50
noise_level_in_delta = 30


def create_a_batch_from_image(image_tf):
    # batch_size = (int(image_tf.shape[0]) / window_size) * (int(image_tf.shape[1]) / window_size)
    image_shape = tf.shape(image_tf)
    batch_size = (image_shape[0] / window_size) * (image_shape[1] / window_size)
    # batch_size = (int(image_tf.shape[0]) / window_size) * (int(image_tf.shape[1]) / window_size)
    batch = tf.convert_to_tensor(np.empty([1, window_size, window_size, number_of_channels]), dtype=tf.uint8)
    print("batch_size = ", batch_size)
    counter = 0
    for i in range(int(int(image_tf.shape[0]) / window_size)):
        for j in range(int(int(image_tf.shape[1]) / window_size)):
            left_up_corner_x = i * (int(int(image_tf.shape[0]) / window_size))
            left_up_corner_y = j * (int(int(image_tf.shape[1]) / window_size))
            current_window = image_tf[left_up_corner_x: left_up_corner_x + window_size,
                             left_up_corner_y: left_up_corner_y + window_size, :]
            print("i = ", i, "j = ", j, " current_window ", current_window.shape,
                  "left_up_corner_x = ", left_up_corner_x, " left_up_corner_y = ", left_up_corner_y)
            batch = tf.concat([batch, [current_window]], 0)
            print("batch ", batch.shape)

        batch_index = i * (int(int(image_tf.shape[0]) / window_size)) + 25
        print("first image number = ", batch_index, " ", batch[batch_index])
        first_image_batch = tf.image.encode_jpeg(batch[batch_index])
        # ,"rgb", 100, False, False, True, "in", 72, 72)
        print("encoded first_image_batch ", first_image_batch)
        writeToNatasha = tf.write_file("./natsha_test" + str(i) + ".jpeg",
                                       first_image_batch, "writeToNatasha" + str(i))
        with tf.Session() as session:
            _, runtime_batch_size = session.run([writeToNatasha, batch_size])
            print('runtime batch size = ', runtime_batch_size)


def dist(rgb, etalon):
    return np.sqrt(np.sum((rgb - etalon) * (rgb - etalon)))


# if   204 < r        and       g < 26   and     b <    29: # RED
# elif 220 < r        and 204 < g        and     b <    25: # MAGENTA
# elif   6 < r <  64  and       g <  52  and 156 < b < 199: # GREEN
# elif       r <  78  and 124 < g < 221  and  31 < b <  85: # BLUE
# elif  59 < r < 115  and       g <  49  and  19 < b <  80: # BROWN

def is_color(rgb, color):
    if color == 'red':
        etalon = np.array([247, 8, 3])   # 1 red: adult males
    if color == 'magenta':
        etalon = np.array([221, 4, 226]) # 2 magenta: subadult males
    if color == 'brown':
        etalon = np.array([88, 46, 8])   # 3 brown: adult females
    if color == 'blue':
        etalon = np.array([31, 65, 189]) # 4 blue: juveniles
    if color == 'green':
        etalon = np.array([44, 178, 21]) # 5 green: pups
    if color == 'black':
        etalon = np.array([0, 0, 0])     # black

    if dist(rgb, etalon) < color_delta:
            return True
    else:
        return False


# train_id,adult_males,subadult_males,adult_females,juveniles,pups
# 41,15,0,85,18,59
# 42,7,4,10,1,0
# 43,28,4,338,47,189
# 44,3,2,25,15,0
# 45,4,7,100,27,0
# 46,1,4,0,0,0
# 47,13,16,48,3,33
# 48,5,10,66,24,0
# 49,0,0,4,15,0
# 50,1,0,0,0,0

# 1 red: adult males
# 2 magenta: subadult males
# 3 brown: adult females
# 4 blue: juveniles
# 5 green: pups
def add_labels_to_image(image_data, image_dot_data):
    width = image_dot_data.shape[0]
    height = image_dot_data.shape[1]
    empty_labels_layer = np.zeros((width, height, 1))
    new_image_data = np.append(image_data, empty_labels_layer, 2)
    print("image_data.shape after append = ", new_image_data.shape)
    print("np.max(image_data) = ", np.max(image_data))
    print("np.max(image_dot_data) = ", np.max(image_dot_data))
    delta = np.abs(image_dot_data - image_data)
    print("np.max(delta) = ", np.max(delta))
    counter_red      = 0
    counter_magenta  = 0
    counter_brown    = 0
    counter_blue     = 0
    counter_green    = 0
    counter_non_zero = 0
    counter_black    = 0
    for i in range(width):
        for j in range(height):
#    for i in range(3100, 3700):
#        for j in range(2800, 3400):
            if is_color(image_dot_data[i][j], 'black'):
                counter_black = counter_black + 1
                new_image_data[i][j][0] = 0
                new_image_data[i][j][1] = 0
                new_image_data[i][j][2] = 0
            else:
                if np.linalg.norm(delta[i][j]) >= noise_level_in_delta:
                    counter_non_zero = counter_non_zero + 1
                    if is_color(image_dot_data[i][j], 'red'):
                        new_image_data[i][j][3] = 1
                        counter_red = counter_red + 1
                    if is_color(image_dot_data[i][j], 'magenta'):
                        new_image_data[i][j][3] = 2
                        counter_magenta = counter_magenta + 1
                    if is_color(image_dot_data[i][j], 'brown'):
                        new_image_data[i][j][3] = 3
                        counter_brown = counter_brown + 1
                    if is_color(image_dot_data[i][j], 'blue'):
                        new_image_data[i][j][3] = 4
                        counter_blue = counter_blue + 1
                    if is_color(image_dot_data[i][j], 'green'):
                        new_image_data[i][j][3] = 5
                        counter_green = counter_green + 1
    s_dot = np.pi * 16.0
    print("counter_red     = ", counter_red     , "approx count = ", counter_red     / s_dot)
    print("counter_magenta = ", counter_magenta , "approx count = ", counter_magenta / s_dot)
    print("counter_brown   = ", counter_brown   , "approx count = ", counter_brown   / s_dot)
    print("counter_blue    = ", counter_blue    , "approx count = ", counter_blue    / s_dot)
    print("counter_green   = ", counter_green   , "approx count = ", counter_green   / s_dot)
    print("counter_black   = ", counter_black   )
    print("counter_non_zero = ", counter_non_zero,
          " black = ", counter_non_zero -
          (counter_red + counter_magenta + counter_brown + counter_blue + counter_green))
    #plt.imshow
    #plt.imsave("natasha" + "43small-test", new_image_data[3100:3700, 2800:3400, :])
    #plt.imsave("natasha" + "43deltasmall-test", delta[3100:3700, 2800:3400, :])
    #print("before save new_image_data ", new_image_data.shape, new_image_data)
    #np.save("testarray43", new_image_data)
    #natasha_test_array_load = np.load("testarray43.npy")
    #print(" after load natasha_test_array_load ", natasha_test_array_load.shape, natasha_test_array_load )
    return new_image_data


def load_picture_from_folder(folder, folder_dot, picture_number):
    image_file = os.path.join(folder, picture_number)
    print(image_file)
    image_dot_file = os.path.join(folder_dot, picture_number)
    print(image_dot_file)
    image_data = []

    try:
        print(datetime.datetime.now())
        image_data = plt.imread(image_file).astype(int)
        print(image_data.shape)
        image_dot_data = plt.imread(image_dot_file).astype(int)
        print(image_dot_data.shape)
        image_data = add_labels_to_image(image_data, image_dot_data)
        np.save("natasha" + picture_number, image_data)
    except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    return image_data


def load_pictures_from_folder(folder, folder_dot):
    image_files = os.listdir(folder)
    print(folder)
    num_images = 0
    for picture_number in image_files:
        try:
            image_data = load_picture_from_folder(folder, folder_dot, picture_number)
        except IOError as e:
            print('Could not read:', picture_number, ':', e, '- it\'s ok, skipping.')


load_pictures_from_folder(path_to_train_small, path_to_train_dot_small)



# if __name__ == '__main__':
#    main()
