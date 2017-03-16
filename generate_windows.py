from skimage.feature import hog
from generate_features import hog_params
import cv2
import numpy as np

def generate_feature_indices(shape, patch_shape=(64, 64),
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
    nblocks_in_patch_y = (ncells_in_patch_y - step_size) + 1
    nblocks_in_patch_x = (ncells_in_patch_x - step_size) + 1

    ncells_y_total = (shape[0]//pix_per_cell[0])  # number of continguous groupings of pixels into cells
    n_blocks_total_y = (ncells_y_total - cell_per_block[0]) + 1  # number of contiguous groupings of pixels into blocks
                                                                 # when you slide by 1

    ncells_x_total = (shape[1]//pix_per_cell[1])
    n_blocks_total_x = (ncells_x_total - cell_per_block[1]) + 1

    adjust_y = ncells_in_patch_y - nblocks_in_patch_y
    adjust_x = ncells_in_patch_x - nblocks_in_patch_x

    for y_idx in range(0, n_blocks_total_y-nblocks_in_patch_y+1, step_size):  # number of patches you have with
        for x_idx in range(0, n_blocks_total_x-nblocks_in_patch_x+1, step_size):  # step_size < patchsize
            adjust_y_current = 0 #adjust_y * y_idx//8                                # note that step size taken in blocks
            adjust_x_current = 0 #adjust_x * x_idx//8
            #print(adjust_y_current)

            y_start = y_idx + adjust_y_current
            x_start = x_idx + adjust_x_current
            indices_y_hog = slice(y_start, y_start + nblocks_in_patch_y)
            indices_x_hog = slice(x_start, x_start + nblocks_in_patch_x)
            indices_y_original = slice(y_idx*pix_per_cell[0], (y_idx+ncells_in_patch_y)*pix_per_cell[0])
            indices_x_original = slice(x_idx*pix_per_cell[1], (x_idx+ncells_in_patch_x)*pix_per_cell[1])
            yield indices_y_hog, indices_x_hog, indices_y_original, indices_x_original


def draw_rectangle(img, indices_y_original, indices_x_original):
    top_left = (indices_x_original.start, indices_y_original.start)
    bottom_right = (indices_x_original.stop, indices_y_original.stop)
    cv2.rectangle(img, top_left, bottom_right, color=(0, 0, 255), thickness=1)


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

def update_heatmap(heatmap, centroids, scale):
    """
    taking the results of applying the classifier to a bunch of regions ,we transform the centroids
    back to their location in the original space and increment relevant entries in the heat map
    :param heatmap:
    :param centroids:
    :param scale:
    :return:
    """
    for point in centroids:
        heatmap[np.int8(point)*scale] += 1

def find_centroids(listofscales, y_start, y_stop):
    pass

if __name__ == '__main__':
    img = cv2.imread('test_images/test1.jpg')
    print(img.shape)
    hog
    hog_features = hog(img[:,:,0], orientations=9,cells_per_block=(2,2), feature_vector=False)
    print(hog_features.shape)
    for indices in generate_feature_indices(shape=(256,256), cell_per_block=(2,2)):
        print(indices)

    """
    draw the sampled square regions
    accum_img = np.zeros((256,1280,3))

    for indices in generate_feature_indices((256,1280)):
        indices_y_hog, indices_x_hog, indices_y_original, indices_x_original = indices
        draw_rectangle(accum_img, indices_y_original=indices_y_original, indices_x_original=indices_x_original)
    cv2.imshow('img', accum_img)
    cv2.waitKey()
    """

