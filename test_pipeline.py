import cv2
from generate_features import show_histos_color_features
from moviepy.editor import VideoFileClip
from normalize_process_images import to_RGB
import numpy as np
from generate_features import get_hog_features
from generate_windows import draw_all_detected_vehicles2, process_heatmap, draw_labeled_bboxes
from scipy.ndimage.measurements import label

def pipeline(img):
    pass

def insert_diag_into(frame, diag, x_slice, y_slice):
    # should take in upper left and lower right pixel location
    x_shape = x_slice.stop - x_slice.start
    y_shape = y_slice.stop - y_slice.start
    frame[x_slice, y_slice] = cv2.resize(to_RGB(diag), (y_shape, x_shape), interpolation=cv2.INTER_AREA)

def compose_diag_screen(curverad=0, offset=0, main_diag=None,
                        diag1=None, diag2=None, diag3=None, diag4=None,
                        diag5=None, diag6=None, diag7=None, diag8=None,
                        diag9=None, diag10=None, diag11=None, diag12=None):
    #  middle panel text example
    #  using cv2 for drawing text in diagnostic pipeline.
    font = cv2.FONT_HERSHEY_COMPLEX
    middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
    cv2.putText(middlepanel, 'Estimated lane curvature: {}'.format(curverad), (30, 60), font, 1, (255, 0, 0), 2)
    cv2.putText(middlepanel, 'Estimated Meters right of center: {}'.format(offset), (30, 90), font, 1, (255, 0, 0), 2)

    # frame that contains all altered images
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # contains slices which describe placement within screen
    # first is x_slice, second y_slice
    # both slices define the shape of the image to be placed
    placement_diag = [(slice(0, 720), slice(0, 1280), main_diag),
                      (slice(0, 240), slice(1280, 1600), diag1),
                      (slice(0, 240), slice(1600, 1920), diag2),
                      (slice(240, 480), slice(1280, 1600), diag3),
                      (slice(240, 480), slice(1600, 1920), diag4),

                      (slice(600, 840), slice(1280, 1600), diag5),
                      (slice(600, 840), slice(1600, 1920), diag6),
                      (slice(840, 1080), slice(1280, 1600), diag7),

                      (slice(840, 1080), slice(1600, 1920), diag8),

                      (slice(720, 840), slice(0, 1280), middlepanel),
                      (slice(840, 1080), slice(0, 320), diag9),
                      (slice(840, 1080), slice(320, 640), diag10),
                      (slice(840, 1080), slice(640, 960), diag11),
                      (slice(840, 1080), slice(960, 1280), diag12)]

    # place all diags within frame
    for x_slice, y_slice, diag in placement_diag:
        if diag is not None:
            insert_diag_into(frame, diag, x_slice, y_slice)

    return frame

def run_compose_diag_screen(img):
    curverad, offset, framed_lane, processing_steps = testing_pipeline(img)
    result = compose_diag_screen(curverad, offset, framed_lane, **processing_steps)
    return result


def testing_pipeline(img):
    #get_rgb = show_histos_color_features()
    #histo_0, histo_1, histo_2 = get_rgb(img)
    #features, hog_img = get_hog_features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), orient=8, pix_per_cell=20, cell_per_block=5, vis=True)

    w_boxes, heatmap = draw_all_detected_vehicles2(img)
    heatmap = process_heatmap(heatmap)
    labels = label(heatmap)
    labeled_boxes = draw_labeled_bboxes(img, labels)

    processing_steps = {
        'diag1': w_boxes,
        #'diag2': histo_1,
        #'diag3': histo_2,
        #'diag4': hog_img,
        #'diag5': img[:,:,0],
        #'diag6': img[:,:,1],
        #'diag7': img[:,:,2],
        #'diag8': convolution,
        #'diag9': all_hough_lines_img,
        #'diag10': filtered_line_image,
        #'diag11': convergence_line_image,
        #'diag12': hull_lines_img
    }
    return 0, 0, labeled_boxes, processing_steps

def run_pipeline(img):
    curverad, offset, main_img, processing_steps = testing_pipeline(img)
    return compose_diag_screen(curverad, offset, main_img, **processing_steps)


if __name__ == '__main__':
    output_path = 'test_project.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(run_pipeline)  # NOTE: this function expects color images!!
    white_clip.write_videofile(output_path, audio=False)
