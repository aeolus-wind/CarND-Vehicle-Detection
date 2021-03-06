import cv2
from generate_features import show_histos_color_features
from moviepy.editor import VideoFileClip
from normalize_process_images import to_RGB
import numpy as np
from generate_features import get_hog_features
from generate_windows import draw_all_detected_vehicles2, process_heatmap, draw_labeled_bboxs
from scipy.ndimage.measurements import label
from car_tracker import CarTracker
from generate_windows import get_bboxs, draw_bboxs

tracker = CarTracker()

def pipeline(img):
    pass

def insert_diag_into(frame, diag, x_slice, y_slice):
    # should take in upper left and lower right pixel location
    x_shape = x_slice.stop - x_slice.start
    y_shape = y_slice.stop - y_slice.start
    frame[x_slice, y_slice] = cv2.resize(to_RGB(diag), (y_shape, x_shape), interpolation=cv2.INTER_AREA)

def compose_diag_screen(main_diag=None,
                        diag1=None, diag2=None, diag3=None, diag4=None,
                        diag5=None, diag6=None, diag7=None, diag8=None,
                        diag9=None, diag10=None, diag11=None, diag12=None):
    #  middle panel text example
    #  using cv2 for drawing text in diagnostic pipeline.
    middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)

    # frame that contains all altered images
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # contains slices which describe placement within screen
    # first is x_slice, second y_slice
    # both slices define the shape of the image to be placed
    placement_diag = [(slice(0, 720), slice(0, 1280), main_diag),
                      (slice(0, 480), slice(1280, 1920), diag1),

                      (slice(600, 1080), slice(1280, 1920), diag5),
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
    global tracker

    w_boxes, heatmap = draw_all_detected_vehicles2(img)
    heatmap = process_heatmap(heatmap)
    labels = label(heatmap)
    raw_bboxs = get_bboxs(labels)

    tracker.update_cars(raw_bboxs)
    filtered_bboxs = tracker.get_bboxs()

    draw_filtered_bboxs = draw_bboxs(img, filtered_bboxs)
    labeled_boxes = draw_labeled_bboxs(img, labels)

    cv2.line(labeled_boxes, (0, 400), (1280, 400), (0,0,255), thickness=3)
    cv2.line(labeled_boxes, (0, 656), (1280, 656), (0, 0, 255), thickness=3)

    processing_steps = {
        'diag1': w_boxes,
        'diag5': labeled_boxes,
        #'diag6': img[:,:,1],
        #'diag7': img[:,:,2],
        #'diag8': convolution,
        #'diag9': all_hough_lines_img,
        #'diag10': filtered_line_image,
        #'diag11': convergence_line_image,
        #'diag12': hull_lines_img
    }
    return draw_filtered_bboxs, processing_steps

def run_pipeline(img):
    main_img, processing_steps = testing_pipeline(img)
    return compose_diag_screen(main_img, **processing_steps)


if __name__ == '__main__':
    output_path = 'test_project.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(run_pipeline)  # NOTE: this function expects color images!!
    white_clip.write_videofile(output_path, audio=False)
