from skimage.feature import hog
from generate_features import hog_params, color_scheme, get_hog_features, bin_image, hist_features, read_color_histos
from train_model import process_data_train_model
import cv2
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
from normalize_process_images import to_RGB


def generate_feature_indices(shape, scale,  patch_shape=(64, 64),
                             pix_per_cell=(8,8), cell_per_block=(2,2), step_size=2):
    """
    This function captures the dimension change that occurs between hog and the other class of features that we
    generate, making the one-to-one mapping between the source of the features evident. The returned values
    are the generated indices.
    :param patch_shape: default size of the raw training data
    :param pixels_per_cell: the default value recommended in the lesson. This breaks up a patch into 8x8 cells
    :param cell_per_block: the default value recommended in the lesson. 4 cells are grouped into a block. pixels are
    reused. The first non-trivial case is illuminated by the square below. If each square is a pixel, and we say each
    pixel is a cell (i.e. pixels_per_cell=(1,1)), and the cells_per_bock is (2,2), we have 3 possible blocks. We only
    get all of these blocks if we say that the step size is 1. This would give us an overlap of 50% in the
    terminology of the lesson. If the step size is 2, we get 0 overlap.

    When you increase the index, step size is how many cells you move. When you access indices, can you access cells
    or do you access blocks?
     ____ ____ ____ ____
    |    |    |    |    |
    |____|____|____|____|
    |    |    |    |    |
    |____|____|____|____|
    :return:
    """
    assert patch_shape[0] % pix_per_cell[0] == 0
    assert patch_shape[1] % pix_per_cell[1] == 0
    ncells_in_patch_y = int(patch_shape[0]/pix_per_cell[0])
    ncells_in_patch_x = int(patch_shape[1]/pix_per_cell[1])
    nblocks_in_patch_y = (ncells_in_patch_y - cell_per_block[0]) + 1
    nblocks_in_patch_x = (ncells_in_patch_x - cell_per_block[1]) + 1

    ncells_y_total = (shape[0]//pix_per_cell[0])  # number of continguous groupings of pixels into cells
    n_blocks_total_y = (ncells_y_total - cell_per_block[0]) + 1  # number of contiguous groupings of pixels into blocks
                                                                 # when you slide by 1

    ncells_x_total = (shape[1]//pix_per_cell[1])
    n_blocks_total_x = (ncells_x_total - cell_per_block[1]) + 1

    adjust_y = ncells_in_patch_y - nblocks_in_patch_y
    adjust_x = ncells_in_patch_x - nblocks_in_patch_x

    for y_idx in range(0, n_blocks_total_y-nblocks_in_patch_y+1, step_size):  # number of patches you have with
        for x_idx in range(0, n_blocks_total_x-nblocks_in_patch_x+1, step_size):  # step_size < patchsize
            indices_y_hog = slice(y_idx, y_idx + nblocks_in_patch_y)
            indices_x_hog = slice(x_idx, x_idx + nblocks_in_patch_x)
            #print(y_idx)
            indices_y_original = slice(y_idx*pix_per_cell[0], (y_idx+ncells_in_patch_y)*pix_per_cell[0])
            indices_x_original = slice(x_idx*pix_per_cell[1], (x_idx+ncells_in_patch_x)*pix_per_cell[1])
            yield indices_y_hog, indices_x_hog, indices_y_original, indices_x_original


def draw_rectangle(img, indices_y_original, indices_x_original):
    top_left = (indices_x_original.start, indices_y_original.start)
    bottom_right = (indices_x_original.stop, indices_y_original.stop)
    cv2.rectangle(img, top_left, bottom_right, color=(0, 0, 255), thickness=2)


def show_all_generated_zones(img_size):
    """
    generates blank images with all regions from which values are sampled for use with the classifier
    :param img_size:
    :return: an image with all the drawn rectangles
    """
    accum_img = np.zeros(img_size)
    for indices in generate_feature_indices(img_size):
        indices_y_hog, indices_x_hog, indices_y_original, indices_x_original = indices
        draw_rectangle(accum_img, indices_y_original=indices_y_original, indices_x_original=indices_x_original)
    return accum_img

def update_heatmap(heatmap, boxes, scale):
    """
    taking the results of applying the classifier to a bunch of regions ,we transform the centroids
    back to their location in the original space and increment relevant entries in the heat map
    :param heatmap:
    :param centroids:
    :param scale:
    :return:
    """
    for y_slice, x_slice in boxes:
        heatmap[y_slice, x_slice] += 1

def find_centroids(img, scale, heatmap_shape, y_start, y_stop, trained_model, step_size, normalize, is_cv2=False):
    region_interested_img = img[y_start: y_stop, :, :]
    heatmap = np.zeros(heatmap_shape)  # heatmap at original scale
    shape = region_interested_img.shape
    if scale != 1:
        shape = (int(shape[0]/scale), int(shape[1]/scale))
        region_interested_img = cv2.resize(region_interested_img, (shape[1], shape[0]))

    if is_cv2:
        rgb = color_scheme(region_interested_img)
        gray = cv2.cvtColor(region_interested_img, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(region_interested_img, cv2.COLOR_BGR2HLS)
    else:
        rgb = img
        gray = cv2.cvtColor(region_interested_img, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(region_interested_img, cv2.COLOR_RGB2HLS)

     #color_scheme(region_interested_img, cspace='HLS')
    batch_hog_features_gray = get_hog_features(gray, **hog_params, feature_vec=False)
    batch_hog_features_h = get_hog_features(hls[:, :, 0], **hog_params, feature_vec=False)
    batch_hog_features_s = get_hog_features(hls[:, :, 2], **hog_params, feature_vec=False)


    boxes = []
    centroids = []
    for indices in generate_feature_indices(shape, scale, step_size=step_size):
        indices_y_hog, indices_x_hog, indices_y_original, indices_x_original = indices

        bin_features = bin_image(rgb, shape=(20, 20))
        rgb_hist_features = hist_features(*read_color_histos(rgb[indices_y_original, indices_x_original], bins=32, range=(0,256)))
        hls_hist_features = hist_features(*read_color_histos(hls[indices_y_original, indices_x_original], bins=32, range=(0,256)))
        hog_features_gray = batch_hog_features_gray[indices_y_hog, indices_x_hog].ravel()
        hog_features_h = batch_hog_features_h[indices_y_hog, indices_x_hog].ravel()
        hog_features_s = batch_hog_features_s[indices_y_hog, indices_x_hog].ravel()
        features = np.concatenate((bin_features, rgb_hist_features, hls_hist_features, hog_features_gray, hog_features_h, hog_features_s)).reshape((1,-1))
        features = normalize.transform(features)


        if int(trained_model.predict(features)[0]) == 1:   # predicted a car
            if scale != 1:
                #shift/scale y and x to original space
                y = slice(y_start + int(indices_y_original.start*scale), y_start + int(indices_y_original.stop*scale))
                x = slice(int(indices_x_original.start*scale), int(indices_x_original.stop*scale))
                boxes.append((y, x))
            else:
                y = slice(y_start + int(indices_y_original.start*scale), y_start + int(indices_y_original.stop*scale))
                boxes.append((y, indices_x_original))
    update_heatmap(heatmap, boxes, scale)

    return heatmap, boxes




def draw_all_detected_vehicles(img):
    model = joblib.load('model.pkl')
    normalize = joblib.load('normalize.pkl')
    all_boxes = []
    for scale in [1.0]:
        heatmap, boxes = find_centroids(img, scale=scale, y_start=400, y_stop=656, trained_model=model, step_size=4, normalize=normalize)
        all_boxes += boxes

    blank = np.zeros_like(img)
    for y, x in all_boxes:
        y = slice(y.start + 400, y.stop + 400)  # shift down
        draw_rectangle(blank, y, x)
    w_boxes = cv2.addWeighted(blank, 1.0, img, 1.0, 0)
    return w_boxes


def draw_all_detected_vehicles2(img):
    model=joblib.load('model.pkl')
    normalize = joblib.load('normalize.pkl')
    heatmap_shape = img.shape[:2]
    all_boxes = []
    heatmap_accum = np.zeros(heatmap_shape)
    heatmap, boxes = find_centroids(img, scale=1.7, heatmap_shape=heatmap_shape, y_start=400, y_stop=656, trained_model=model, step_size=3,
                                    normalize=normalize)
    all_boxes += boxes
    heatmap_accum += heatmap

    constant_parameters = {'normalize': normalize,
                           'trained_model': model,
                           'heatmap_shape': heatmap_shape
                           }
    variable_parameters = [
        {'scale': 2.0,
         'y_start': 400,
         'y_stop': 656,
         'step_size': 3
         },
        {'scale': 2.0,
         'y_start': 400,
         'y_stop': 528,
         'step_size': 3
         },
        {'scale': 160/64.0,
         'y_start': 520,
         'y_stop':680,
         'step_size':3
         },
        {'scale':0.5,
         'y_start':350,
         'y_stop': 414,
         'step_size': 4
        }
    ]

    for vp in variable_parameters:
        heatmap, boxes = find_centroids(img, **vp, **constant_parameters)
        all_boxes += boxes
        heatmap_accum += heatmap

    blank = np.zeros_like(img)
    for y, x in all_boxes:
        draw_rectangle(blank, y, x)
    w_boxes = cv2.addWeighted(blank, 1.0, img, 1.0, 0)
    return w_boxes, heatmap_accum


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


# regions which are excluded from heatmap
# the rational is that these deal with separate subproblems
# if a car is in the too_close_to_car region
# the car will have to perform some sort of evasive maneuver
# we will also see the other car merging in from the other lane or appear from in front or beside us
# We also note that this case is distinguished by the fact the labeling box will be very large
# these cases are outside the scope of this project
# the left of lane mask should be dynamic and use a setup like the last project
# for the purposes of this, it should be fine that it's a constant region
too_close_to_car = [np.array([(500, 520), (291, 690), (1080, 680), (812, 520)], np.int32)]
left_of_lane = [np.array([(0,691), (0,400), (646,400), (291, 691)], np.int32)]


def remove_cases_from_heatmap(heatmap, region_to_remove):
    if len(heatmap.shape)<2:
        print("heat map wrong dimension")
    cv2.fillPoly(heatmap, region_to_remove, (0, 0, 0))


def process_heatmap(heatmap):
    remove_cases_from_heatmap(heatmap, too_close_to_car)
    remove_cases_from_heatmap(heatmap, left_of_lane)
    return apply_threshold(heatmap, 2)

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

if __name__ == '__main__':
    #generate_feature_indices((64,64), scale, patch_shape=(64, 64),
#                             pix_per_cell=(8, 8), cell_per_block=(2, 2), step_size=2)

    #model, normalize = process_data_train_model(0.1, 'model', 'normalize')
    img = cv2.imread('test_images/test1.jpg')

    w_boxes, heatmap = draw_all_detected_vehicles2(img)
    heatmap = process_heatmap(heatmap)
    labels = label(heatmap)
    labeled_boxes = draw_labeled_bboxes(img, labels)


    cv2.imshow('img', labeled_boxes)
    cv2.waitKey()

    """
    all_boxes = []
    for scale in [1.0]:
        heatmap, boxes = find_centroids(img, scale=scale, y_start=400, y_stop=656, trained_model=model, step_size=2)
        all_boxes += boxes

    blank = np.zeros_like(img)
    for y, x in all_boxes:
        y = slice(y.start+400, y.stop + 400) #shift down
        draw_rectangle(blank, y, x)
    w_boxes = cv2.addWeighted(blank, 1.0, img, 1.0, 0)
    cv2.imshow('w_boxes', w_boxes)
    cv2.waitKey()
    """
    #plt.imshow(cv2.cvtColor(w_boxes,cv2.COLOR_BGR2RGB))
    #plt.show()



    """
    img = cv2.imread('test_images/test1.jpg')
    print(img.shape)
    hog_features = hog(img[:170,:853,0], orientations=9,cells_per_block=(2,2), feature_vector=False)
    print(hog_features.shape)
    for indices in generate_feature_indices(shape=(170,853), cell_per_block=(2, 2), step_size=2):
        print(indices)
    """

    """
    draw the sampled square regions
    accum_img = np.zeros((256,1280,3))

    for indices in generate_feature_indices((256,1280)):
        indices_y_hog, indices_x_hog, indices_y_original, indices_x_original = indices
        draw_rectangle(accum_img, indices_y_original=indices_y_original, indices_x_original=indices_x_original)
    cv2.imshow('img', accum_img)
    cv2.waitKey()
    """
    """
    img = cv2.imread('test_images/test1.jpg')
    accum_img = np.zeros((720, 1280, 3))

    for indices in generate_feature_indices((256, 1280)):
        indices_y_hog, indices_x_hog, indices_y_original, indices_x_original = indices
        y = slice(indices_y_original.start + 400, indices_y_original.stop + 400)  # shift down

        draw_rectangle(accum_img, indices_y_original=y, indices_x_original=indices_x_original)

    w_boxes = cv2.addWeighted(np.uint8(accum_img), 0.5, np.uint8(img), 1.0, 0)
    cv2.imshow('img', w_boxes)
    cv2.waitKey()
    """


