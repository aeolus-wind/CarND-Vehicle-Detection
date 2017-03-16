from skimage.feature import hog
from generate_features import hog_params
import cv2

def generate_all_hog_cells(img, y_start, y_stop, hog_params, scale=1):
    """
    takes in a 2-d image and generates all hog feature rectangles
    :param img:
    :param y_lower:
    :param y_upper:
    :param filter_size: restricted to be symmetric in x, y direction
    :param step_size: restricted to be symmetric in x, y direction
    :return:
    """
    region_interest = img[y_start: y_stop, :]
    shape = region_interest.shape
    if scale != 1:
        region_interest = cv2.resize(region_interest, int(shape[0]/scale), int(shape[1]/scale))
    shape = region_interest.shape
    total_blocks_x = shape[1]//hog_params['pix_per_cell'] - 1
    total_blocks_y = shape[0]//hog_params['pix_per_cell'] - 1
    features_in_block = hog_params['orient'] * hog_params['cell_per_block']**2
    window = 64
    cells_per_step = 2
    blocks_per_window = window // hog_params['pix_per_cell'] - 1
    x_steps = (total_blocks_x - blocks_per_window)//cells_per_step
    y_steps = (total_blocks_y - blocks_per_window)//cells_per_step

    all_hog_features = hog(region_interest, **hog_params, feature_vector=False)
    hog_window_features = []
    for x_step in range(x_steps):
        for y_step in range(y_steps):
            x_pos = x_step*cells_per_step
            y_pos = y_step * cells_per_step
            hog_window_features.append(all_hog_features[y_pos: y_pos+blocks_per_window,
                                                        x_pos: x_pos + blocks_per_window].ravel())
            x_left = x_step * hog_params['pix_per_cell']
            y_top = y_step * hog_params['pix_per_cell']

            subimg = cv2.resize(region_interest[y_top: y_top + window, x_left: x_left + window], (64, 64))
            bin_features = bin_image(subimg, (20, 20))
            hist_features = hist_features(read_color_histos(subimg))


def generate_feature_indices(shape, patch_shape=(64, 64),
                             pixels_per_cell=(8,8), cell_per_block=(2,2), step_size=2):
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
    assert patch_shape[0] % pixels_per_cell[0] == 0
    assert patch_shape[1] % pixels_per_cell[1] == 0
    ncells_in_patch_y = int(patch_shape[0]/pixels_per_cell[0])
    ncells_in_patch_x = int(patch_shape[1]/pixels_per_cell[1])
    nblocks_in_patch_y = (ncells_in_patch_y - step_size) + 1
    nblocks_in_patch_x = (ncells_in_patch_x - step_size) + 1

    ncells_y_total = (shape[0]//pixels_per_cell[0])  # number of continguous groupings of pixels into cells
    n_blocks_total_y = (ncells_y_total - cell_per_block[0]) + 1  # number of contiguous groupings of pixels into blocks
                                                                 # when you slide by 1

    ncells_x_total = (shape[1]//pixels_per_cell[1])
    n_blocks_total_x = (ncells_x_total - cell_per_block[1]) + 1

    adjust_y = ncells_in_patch_y - nblocks_in_patch_y
    adjust_x = ncells_in_patch_x - nblocks_in_patch_x

    for y_idx in range(0, n_blocks_total_y-nblocks_in_patch_y+1, step_size):  # number of patches you have with
        for x_idx in range(0, n_blocks_total_x-nblocks_in_patch_x+1, step_size):  # step_size < patchsize
            adjust_y_current = adjust_y * y_idx//8                                # note that step size taken in blocks
            adjust_x_current = adjust_x * x_idx//8

            y_start = y_idx + adjust_y_current
            x_start = x_idx + adjust_x_current
            indices_y_hog = slice(y_start, y_start + nblocks_in_patch_y)
            indices_x_hog = slice(x_start, x_start + nblocks_in_patch_x)
            indices_y_original = slice(y_idx*pixels_per_cell[0], (y_idx+ncells_in_patch_y)*pixels_per_cell[0])
            indices_x_original = slice(x_idx*pixels_per_cell[1], (x_idx+ncells_in_patch_x)*pixels_per_cell[1])
            yield indices_y_hog, indices_x_hog, indices_y_original, indices_x_original




    """
    ny_block_center_in_patch = patch_shape[0]//pixels_per_cell[0] - 1
    nx_block_center_in_patch = patch_shape[1]//pixels_per_cell[1] - 1
    ny_patches_image = shape[0]//patch_shape[0] - 1
    nx_patches_image = shape[1]//patch_shape[1] - 1
    ny_cells = shape[0]//pixels_per_cell[0] - 1
    nx_cells = shape[1]//pixels_per_cell[1] - 1
    """

if __name__ == '__main__':
    generate_indices = generate_feature_indices(shape=(128,128), scale=1)
    for possible_index in generate_indices:
        print(possible_index)