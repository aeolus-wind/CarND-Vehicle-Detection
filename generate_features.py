import cv2
import numpy as np
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import glob
from normalize_process_images import to_RGB

hog_params = {'orient': 8, 'pix_per_cell': 8, 'cell_per_block': 2}


def color_scheme(img, cspace='RGB'):
    if cspace == 'RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif cspace == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif cspace == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif cspace == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif cspace == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    else:
        print("color not found in function")
        return img

def bin_image(img, shape=(32,32)):
    return cv2.resize(img, shape).ravel()


def read_color_histos(img, bins=32, range=(0,256)):
    histo_0 = np.histogram(img[:, :, 0], bins=bins, range=range)
    histo_1 = np.histogram(img[:, :, 1], bins=bins, range=range)
    histo_2 = np.histogram(img[:, :, 2], bins=bins, range=range)

    return histo_0, histo_1, histo_2

def hist_features(histo_0, histo_1, histo_2):
    return np.concatenate((histo_0[0], histo_1[0], histo_2[0]))


def convert_histos_to_np(histo_0, histo_1, histo_2):
    """
    http://zulko.github.io/blog/2014/11/29/data-animations-with-python-and-moviepy/
    """
    fig1, ax1 = plt.subplots(1, figsize=(5, 3))
    fig2, ax2 = plt.subplots(1, figsize=(5, 3))
    fig3, ax3 = plt.subplots(1, figsize=(5, 3))
    bin_edges = histo_0[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    ax1.set_title('histogram color coordinate 0')
    ax1.set_xlim(0, 256)
    ax1.bar(bin_centers, histo_0[0])
    ax2.set_title('histogram color coordinate 1')
    ax2.set_xlim(0,256)
    ax2.bar(bin_centers, histo_1[0])
    ax3.set_title('histogram color coordiante 2')
    ax3.set_xlim(0,256)
    ax3.bar(bin_centers, histo_2[0])
    return mplfig_to_npimage(fig1), mplfig_to_npimage(fig2), mplfig_to_npimage(fig3)


def show_histos_color_features(cspace='RGB', bins=32, range=(0,256)):
    def histos_color_features(img):
        img = color_scheme(img, cspace=cspace)
        histo_0, histo_1, histo_2 = read_color_histos(img, bins, range)
        return convert_histos_to_np(histo_0, histo_1, histo_2)
    return histos_color_features



def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orient, (pix_per_cell, pix_per_cell), (cell_per_block, cell_per_block),
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orient, (pix_per_cell, pix_per_cell), (cell_per_block, cell_per_block),
                       visualise=vis, feature_vector=feature_vec)
        return features

def get_features(img):
    hls = color_scheme(img, cspace='HLS')
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yuv = color_scheme(img, 'YUV')

    bin_features = bin_image(yuv, (32,32))
    hls_features = hist_features(*read_color_histos(hls, bins=32, range=(0, 256)))
    yuv_features = hist_features(*read_color_histos(yuv, bins=32, range=(0, 256)))

    hog_features_y = get_hog_features(yuv[:,:,0], **hog_params)
    hog_features_u = get_hog_features(yuv[:, :, 1], **hog_params)
    hog_features_v = get_hog_features(yuv[:, :, 2], **hog_params)
    features = np.concatenate((bin_features, hls_features, yuv_features, hog_features_y, hog_features_u,
                               hog_features_v)).reshape((1, -1))
    return features


def plot_histogram(img, histo_number=1):
    h1, h2, h3 = read_color_histos(img)

    bin_edges = h1[1]
    bin_centers = (bin_edges[1:] + bin_edges[0: -1])/ 2

    plt.figure(figsize=(12,3))
    plt.subplot(131)
    plt.bar(bin_centers, h1[0])
    plt.xlim(0,256)
    plt.title('histogram 1')
    plt.subplot(132)
    plt.bar(bin_centers, h2[0])
    plt.xlim(0, 256)
    plt.title('histogram 2')
    plt.subplot(133)
    plt.bar(bin_centers, h3[0])
    plt.xlim(0, 256)
    plt.title('histogram 3')
    plt.show()




if __name__=='__main__':
    img = cv2.imread('test_images/test1.jpg')
    img = color_scheme(img, cspace='YUV')
    #y, u, v = read_color_histos(img)
    #y, u, v = convert_histos_to_np(y,u,v)
    #plot_histogram(img)

    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    features, hog_img = get_hog_features(luv[:, :, 1], **hog_params, vis=True)

    cv2.imwrite('./writeup_images/hog_example.png', to_RGB(hog_img))
    print(hog_img.shape)

    cv2.imshow('img', hog_img)
    cv2.waitKey()

    """
    all_features = []
    far_images = glob.glob('vehicles/KITTI_extracted/*.png')
    for img_path in far_images:
        img = cv2.imread(img_path)
        all_features.append(get_features(img))
    X = np.concatenate(all_features)
    scale = StandardScaler()
    renormed = scale.fit_transform(X)
    """

