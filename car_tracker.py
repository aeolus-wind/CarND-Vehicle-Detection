import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import math
from generate_windows import draw_all_detected_vehicles2
from generate_windows import process_heatmap
from scipy.ndimage.measurements import label
import unittest
import cv2
from generate_windows import get_bboxs


class CarTracker():
    def __init__(self):
        self.cars = []
        self.cars_wait_counter = []
        self.potential_cars = []
        self.potential_cars_wait_counter = []

    def add_car(self, bbox):
        self.cars.append(Car(bbox))
        self.car_wait_counter.append(0)

    def add_potential_car(self, bbox):
        self.potential_cars.append(Car(bbox))
        self.potential_cars_wait_counter.append(0)

    def get_bboxs(self):
        bboxs = []
        for car in self.cars:
            bboxs.append(car.avg_bbox())
        return bboxs


    def is_emerging(self, bbox):
        """
        observe for 2 frames, velocity must be increasing or decreasing based on region
        :param bbox:
        :return:
        """
        emerging_regions = np.array([((900, 330), (1280, 700)), ((600, 300), (1000, 550))]) # can only #use direction and velocity to determine

        # a tuple where first coordinate shows if the bbox is contained in emerging region
        # second coordinate is if bbox is in the forward emerging region i.e. is the car slowing down
        return (np.all(emerging_regions[0][0] <= bbox[0]) and np.all(emerging_regions[0][1] >= bbox[1]),
               np.all(emerging_regions[1][0] <= bbox[0]) and np.all(emerging_regions[1][1] >= bbox[1]))


    def update_cars(self, bboxs_new):
        """
        to fix: will the noisiness of the labeling algorithm break this? if the box changes shape a lot,
        the algorithm will not accept it. This will lead to many premature deletions due to the 5 observation rule
        solution 1: fix the classifier
        solution 2: play around with the acceptance
        TODO: also right now, everything is just matched, deviation_bbox isn't used
        :param bboxs_new:
        :return:
        """
        # match to current cars
        matches_info, matched = self.match_bboxs(self.cars, bboxs_new)

        not_matched = [i for i in range(len(bboxs_new)) if i not in matched]  # get indices
        bboxs_not_matched_to_current = [bboxs_new[i] for i in not_matched]  # pull out bboxs relevant entries
        entering_matches_info, entering_matched = self.match_bboxs(self.potential_cars, bboxs_not_matched_to_current)
        possible_new = [i for i in range(len(not_matched)) if i not in entering_matched]
        possible_new = [bboxs_not_matched_to_current[i] for i in possible_new]  # possible new bboxs


        for i in range(len(self.cars)):
            if i not in matches_info.keys():
                self.cars_wait_counter[i] += 1  # after bbox has not been updated for a car for a while, force deletion
            else:
                self.cars[i].update_bbox(bboxs_new[matches_info[i][0]]) #update the physics with the new information
                self.cars_wait_counter[i] = 0  # reset counter if car has been observed

        for i in range(len(self.potential_cars)):
            if i not in entering_matches_info.keys():
                self.potential_cars_wait_counter[i] += 1
                """ TODO!!!"""
                # if bbox fails to match for 3 frames, is deleted at end of cycle
            else:
                """TODO!!!"""
                # if physics is updated 5 frames in total, can make a real car
                self.potential_cars[i].update_bbox(bboxs_not_matched_to_current[entering_matches_info[i][0]])
        possible_in_entering = [self.is_emerging(point) for point in possible_new]



        for i, entry in enumerate(possible_in_entering):
            if entry[0] or entry[1]:  # note the entry information can be used if additional info is needed
                self.add_potential_car(possible_new[i])  # for example, look at the direction of velocity

        for i, count_fail_update in reversed(list(enumerate(self.potential_cars_wait_counter))): #delete items that failed to match 3 times
            if count_fail_update >= 3:
                del self.potential_cars_wait_counter[i]
                del self.potential_cars[i]

        for i, count_fail_update_observed_cars in reversed(list(enumerate(self.cars_wait_counter))):
            if count_fail_update_observed_cars >= 15:  # car has not been observed for 5 frames, so delete
                del self.cars[i]
                del self.cars_wait_counter[i]

        for i, test_observations in reversed(list(enumerate(self.potential_cars))):
            if len(self.potential_cars[i].bbox) >= 5:  # potential car has accumulated 5 observations
                self.cars.append(self.potential_cars[i])
                self.cars_wait_counter.append(0)
                del self.potential_cars[i]
                del self.potential_cars_wait_counter[i]


    def match_bboxs(self, cars, bboxs_new):
        """
        :param cars: old observations of bbox e.g. old observations of cars
        :params bboxs_new: new observations-- considering adding to some set
        :return: all matches compared with previous observations
        """
        matches_info = {} # a dictionary which gives the best match to a car where a previous observation of a car
        # where a previous observation is indicated by the index of the car dictionary where it resides
        matched = [] # contains indices of matched to be used in selections later
        if len(cars) > 0:
            for new_car_number, bbox in enumerate(bboxs_new):
                distances = []
                # obtain distances between new observation and all previously labeled car observations
                # old_car number is effectively labeled by index
                for old_car_number, car in enumerate(cars):
                    bbox_predict = car.predict_bbox()  # find distance of new observation with previous car observations

                    distances.append(distance(bbox[0], bbox_predict[0]) + distance(bbox[1], bbox_predict[1]))
                # take all distances to old observations, the smallest is the match
                # this means that each new observation can only match to one old one
                old_car_number_match = np.argmin(distances)
                if old_car_number_match not in matches_info.keys():
                    matches_info[old_car_number_match] = [(new_car_number, np.min(distances))]
                else:
                    matches_info[old_car_number_match].append((new_car_number, np.min(distances)))
            # on the other hand, old car observations may have many new observations matching to them
            # similarly, we look through all these matches_info and select the best (one with smallest distance)
            for old_car_number in matches_info.keys():
                index = np.argmin(np.array(matches_info[old_car_number])[:, 1])  # pick out index of best match
                matches_info[old_car_number] = matches_info[old_car_number][index]
                matched.append(matches_info[old_car_number][0])  # new car number that is completely matched appended
        return matches_info, matched


        #each car can have up to one match
        # the match must make sense

class TestMatchMethods(unittest.TestCase):

    def test_trivial_match_coordinates(self):
        img = mpimg.imread('test_images/test1.jpg')
        w_boxes, heatmap = draw_all_detected_vehicles2(img)
        heatmap = process_heatmap(heatmap)
        labels = label(heatmap)
        bboxs = get_bboxs(labels)
        # currently tinkering with this, but  my current process gives the
        # bboxs [((816, 400), (923, 495)), ((1056, 400), (1239, 519))]
        # this means that the first bbox is closer to the origin
        # since the coordinates below are defined so that car1 is closer to the origin
        # the algorithm will match both potential new cars to previously observed car2
        # of these matches the first bbox at index 0 will be retained because it is closer

        car1 = Car()
        car2 = Car()
        car1.bbox.append(((0.05, 0.05), (1.0, 1.0)))
        car1.update_physics(((0.1, 0.1), (1.1, 1.1)))
        car1.update_physics(((0.2, 0.2), (1.5, 1.5)))
        car2.bbox.append(((2.0, 2.0), (4.0, 4.0)))
        car2.update_physics(((2.5, 2.5), (4.5, 4.5)))
        car2.update_physics(((3.0, 3.5), (5.0, 4.0)))


        test_tracker = CarTracker()
        test_tracker.cars.append(car1)
        test_tracker.cars.append(car2)
        result = test_tracker.match_bboxs(test_tracker.cars, bboxs)
        self.assertTrue(result.keys()[0] == 1 and len(result)==1 and result[1][0][0]==0)

    def test_entering_region_trivial(self):
        test_tracker = CarTracker()
        constant_bbox_test = np.array([((1000, 550), (1100, 650))])
        test_tracker.update_cars(constant_bbox_test)
        test_tracker.update_cars(constant_bbox_test)
        test_tracker.update_cars(constant_bbox_test)
        self.assertNotEqual(test_tracker.potential_cars, [])

    def test_entering_region_deletion(self):

        test_tracker = CarTracker()
        constant_bbox_test = np.array([((1000, 550), (1100, 650))])
        test_tracker.update_cars(constant_bbox_test)
        test_tracker.update_cars(constant_bbox_test)
        test_tracker.update_cars(constant_bbox_test)
        test_tracker.update_cars([])
        test_tracker.update_cars([])
        test_tracker.update_cars([])
        self.assertEqual(test_tracker.potential_cars, [])

    def test_become_valid_car(self):
        test_tracker = CarTracker()
        constant_bbox_test = np.array([((1000, 550), (1100, 650))])
        test_tracker.update_cars(constant_bbox_test)
        test_tracker.update_cars(constant_bbox_test)
        test_tracker.update_cars(constant_bbox_test)
        test_tracker.update_cars(constant_bbox_test)
        self.assertEqual(test_tracker.cars, [])
        self.assertNotEqual(test_tracker.potential_cars, [])
        test_tracker.update_cars(constant_bbox_test)
        self.assertNotEqual(test_tracker.cars, [])
        self.assertEqual(test_tracker.potential_cars, [])






def distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)



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

class Car():
    def __init__(self, maxq=5):
        self.bbox = deque([], maxlen=maxq)
        self.velocity = deque([], maxlen=maxq)
        self.acceleration = deque([], maxlen=maxq)

    def __init__(self, bbox, maxq=5):
        self.bbox = deque([bbox], maxlen=maxq)
        self.velocity = deque([], maxlen=maxq)
        self.acceleration = deque([], maxlen=maxq)


    def update_bbox(self, bbox):
        if self.deviation_bbox(bbox) <= 0.1:

            self.update_physics(bbox)

    def predict_bbox(self):
        if len(self.acceleration) == 0:
            return self.bbox[-1]
        else:
            return np.int32(np.mean(self.bbox, axis=0)) #+ np.int32(np.mean(self.velocity), axis=0) + np.int32(np.mean(self.acceleration), axis=0))

    def avg_bbox(self):
        if len(self.acceleration) == 0:
            return self.bbox[-1]
        else:
            return np.int32(np.mean(self.bbox, axis=0))


    def update_physics(self, bbox):
        """
        :param bbox: vectorized top left and bottom right
        :return:
        """
        bbox = np.array(bbox)
        if len(self.bbox) > 0:
            # if possible to calculate new_velocity, do so
            new_velocity = bbox - self.bbox[-1]
        if len(self.velocity) > 0:
            new_acceleration = new_velocity - self.velocity[-1]
            self.acceleration.append(new_acceleration)

        # updates down here due to dependencies
        if len(self.bbox) > 0:
            self.velocity.append(new_velocity)
        self.bbox.append(bbox)

    def deviation_bbox(self, new_bbox):
        if self.predict_bbox() is None:
            return None
        else:
            new_bbox = np.array(new_bbox)
            prediction = self.predict_bbox()
            deviation = np.mean(np.abs((new_bbox - prediction)) / prediction)
            if np.any(np.isinf(deviation)):
                return None
            else:
                return deviation




if __name__ == '__main__':
    img = mpimg.imread('test_images/test1.jpg')
    w_boxes, heatmap = draw_all_detected_vehicles2(img)
    heatmap = process_heatmap(heatmap)
    labels = label(heatmap)
    bboxs = get_bboxs(labels)


    test_tracker = CarTracker()
    constant_bbox_test = np.array([((1000, 550), (1100, 650))])
    test_tracker.update_cars(constant_bbox_test)
    test_tracker.update_cars(constant_bbox_test)
    test_tracker.update_cars(constant_bbox_test)
    test_tracker.update_cars(constant_bbox_test)
    print(test_tracker.cars)
    print(test_tracker.potential_cars)
    test_tracker.update_cars(constant_bbox_test)
    print(test_tracker.potential_cars)
    print(test_tracker.potential_cars_wait_counter)
    print(test_tracker.cars)
    print(test_tracker.cars[0].avg_bbox())



    #print(test_tracker.is_emerging(np.array(((550,1000), (650, 1100)))))

