from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from generate_features import read_color_histos, get_features
import cv2
import numpy as np
import glob
import pandas as pd
import re
import pickle

model = LinearSVC(C=1.0)

def read_append_features(file_paths, feature_generating_function):
    image_features = []
    for img_path in file_paths:
        img = cv2.imread(img_path)
        image_features.append(feature_generating_function(img))

    non_vehicle_images = np.concatenate(image_features)
    return non_vehicle_images

def read_train_test_split(non_vehicle_root, non_vehicle_folders, vehicle_root, vehicle_folders, test_size=0.33):
    """
    function which takes in a list of folders, randomizes the train/test split after creating the features
    initial version. next version wil have hand-coded time-dependencies added in
    :param folders:
    :return:
    """
    non_vehicle_paths = []
    for folder in non_vehicle_folders:
        non_vehicle_paths += glob.glob(non_vehicle_root + '/' + folder + '/' + '*.png')

    non_vehicle_features = read_append_features(non_vehicle_paths, get_features)

    vehicle_paths = []
    for folder in vehicle_folders:
        vehicle_paths += glob.glob(vehicle_root + '/' + folder + '/' + '*.png')

    vehicle_features = read_append_features(vehicle_paths, get_features)

    X = np.concatenate((non_vehicle_features, vehicle_features), axis=0)
    y = np.hstack((np.zeros(len(non_vehicle_features)), np.ones(len(vehicle_features))))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def get_index_image(full_path):
    file_name = full_path.split('/')[2]
    number_label = re.search('[0-9]+', file_name)
    return int(number_label.group(0))

def in_index(index, list_of_indices):
    for index_bounds in list_of_indices:
        if index >= index_bounds[0] and index <= index_bounds[1]:
            return True
    return False

def check_index_append_train_test(img_paths, test_indice_list, train_list, test_list, feature_generating_function):
    for img_path in img_paths:
        index = get_index_image(img_path)
        img = cv2.imread(img_path)
        features = feature_generating_function(img)
        if in_index(index, test_indice_list):
            test_list.append(features)
        else:
            train_list.append(features)

def shuffle_in_unison(a, b):
    """
    http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    :param a:
    :param b:
    :return:
    """
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def process_time_dependent(non_vehicle_root, vehicle_root, feature_generating_function):
    non_vehicle_train_image_features = []
    non_vehicle_test_image_features = []
    non_vehicle_indices_test = [(1, 175), (248, 556), (557, 767), (1376, 1765), (3651, 3697)]
    non_vehicle_gti_paths = glob.glob(non_vehicle_root + '/' + 'GTI' + '/' + '*.png')

    check_index_append_train_test(non_vehicle_gti_paths, non_vehicle_indices_test, non_vehicle_train_image_features,
                                  non_vehicle_test_image_features, feature_generating_function)

    non_vehicle_train_image_features = np.concatenate(non_vehicle_train_image_features)
    non_vehicle_test_image_features = np.concatenate(non_vehicle_test_image_features)


    vehicle_train_image_features = []
    vehicle_test_image_features = []
    vehicle_indices_test_gti_far = [(500,816)]
    vehicle_indices_test_gti_left = [(9, 285), (583, 772), (773, 974)]
    vehicle_indices_test_gti_middleclose = [(0,247)]
    vehicle_indices_test_gti_right = [(265,535), (771, 974)]

    vehicle_gti_far_paths = glob.glob(vehicle_root + '/' + 'GTI_Far' + '/' + '*.png')
    vehicle_gti_left_paths = glob.glob(vehicle_root + '/' + 'GTI_Left' + '/' + '*.png')
    vehicle_gti_middleclose_paths = glob.glob(vehicle_root + '/' + 'GTI_MiddleClose' + '/' + '*.png')
    vehicle_gti_right_paths = glob.glob(vehicle_root + '/' + 'GTI_Right' + '/' + '*.png')

    check_index_append_train_test(vehicle_gti_far_paths, vehicle_indices_test_gti_far, vehicle_train_image_features,
                                  vehicle_test_image_features, feature_generating_function)
    check_index_append_train_test(vehicle_gti_left_paths, vehicle_indices_test_gti_left, vehicle_train_image_features,
                                  vehicle_test_image_features, feature_generating_function)
    check_index_append_train_test(vehicle_gti_middleclose_paths, vehicle_indices_test_gti_middleclose,
                                  vehicle_train_image_features, vehicle_test_image_features, feature_generating_function)
    check_index_append_train_test(vehicle_gti_right_paths, vehicle_indices_test_gti_right, vehicle_train_image_features,
                                  vehicle_test_image_features, feature_generating_function)

    vehicle_train_image_features = np.concatenate(vehicle_train_image_features)
    vehicle_test_image_features = np.concatenate(vehicle_test_image_features)

    X_train = np.concatenate((non_vehicle_train_image_features, vehicle_train_image_features))
    y_train = np.concatenate((np.zeros(len(non_vehicle_train_image_features)), np.ones(len(vehicle_train_image_features))))
    X_test = np.concatenate((non_vehicle_test_image_features, vehicle_test_image_features))
    y_test = np.concatenate((np.zeros(len(non_vehicle_test_image_features)), np.ones(len(vehicle_test_image_features))))
    # this will be shuffled again when combined with non-time series

    shuffle_in_unison(X_train, y_train)
    shuffle_in_unison(X_test, y_test)
    return X_train, X_test, y_train, y_test





def show_images_in_folder(folder, group_path):
    cv2.namedWindow('image')
    cv2.namedWindow('group_meta')
    images = []
    for i, img_path in enumerate(glob.glob(folder + '/' + '*.png')):
        filename = img_path.split('/')[2]
        img = cv2.resize(cv2.imread(img_path), (500, 500),interpolation=cv2.INTER_AREA)
        blank = np.zeros((100, img.shape[1],3))
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(blank, 'filename: ' + filename, (50, 50), font, 1, (0, 0, 255))
        number_label = re.search('[0-9]+', img_path)
        img = np.uint8(np.concatenate((blank, img), axis=0))
        if number_label:
            images.append((img, int(number_label.group(0))))
        else:
            print('issue with number label')
    counter = 0
    groups = {}
    group_index = 0
    images = pd.DataFrame(images, columns=('images', 'index')).sort_values(by=('index'))

    while True:
        img = images['images'].iloc[counter]
        group_meta_blank = np.zeros((200, 500, 3))
        cv2.putText(group_meta_blank, 'current group is :' + str(group_index), (50, 50), font, 1, (0, 0, 255))


        cv2.imshow('image', img)
        cv2.imshow('group_meta', group_meta_blank)
        key = cv2.waitKey()
        if key == ord('n'):
            counter += 1
        if key == ord('p'):
            if counter > 0:
                counter -= 1
        if key == ord('i'):
            group_index += 1
        if key == ord('d'):
            if group_index > 0:
                group_index -=1
        if key == ord('a'):
            if group_index in groups.keys():
                groups[group_index] += [images['index'].iloc[counter]]
            else:
                groups[group_index] = [images['index'].iloc[counter]]
        if key == ord('x'):
            if len(groups[group_index]) > 0:
                groups[group_index] = groups[group_index][:-1]
        if key == ord('s'):
            if group_index in groups.keys() and len(groups[group_index])>0:
                print('the grouping for index ', str(group_index), ' is ',groups[group_index])
        if key == ord('q'):
            break
    with open(group_path, 'wb') as pickle_file:
        pickle.dump(groups, pickle_file)


if __name__ == '__main__':

    """
    non_vehicle_folders = ['Extras', 'GTI']
    non_vehicle_root = 'non-vehicles'
    vehicle_folders = ['GTI_Far', 'GTI_Left', 'GTI_MiddleClose', 'GTI_Right', 'KITTI_extracted']
    vehicle_root = 'vehicles'
    print(read_train_test_split(non_vehicle_root, non_vehicle_folders, vehicle_root, vehicle_folders, test_size=0.33))
    """
    a = np.array(range(10))
    b = np.array(range(20)).reshape((10,2))
    shuffle_in_unison(a, b)

    #print(process_time_dependent('non-vehicles', 'vehicles', get_features))