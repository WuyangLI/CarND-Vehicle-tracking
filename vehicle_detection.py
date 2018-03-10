import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from vehicle_classifier import get_features
import pickle


def generate_windows(x_start_stop, y_start_stop, window_size, xy_step):
    window_list = []
    x_start = x_start_stop[0]
    x_stop = x_start_stop[1]
    y_start = y_start_stop[0]
    y_stop = y_start_stop[1]
    window_width = window_size[0]
    window_height = window_size[1]
    x_step = int(window_width*xy_step[0])
    y_step = int(window_height*xy_step[1])
    for x in range(x_start, x_stop, x_step):
        for y in range(y_start, y_stop, y_step):
            if x+window_width <= x_stop and y+window_height <= y_stop:
                window_list.append([x, x+window_width, y, y+window_height])
    return window_list


def search_windows(img, svc, orient, pix_per_cell, cell_per_block):
    windows = []
    windows += generate_windows((400, 464), (700, 1280), window_size=(64, 64), xy_step=(0.1, 0.1))
    windows += generate_windows((416, 480), (700, 1280), window_size=(64, 64), xy_step=(0.2, 0.2))
    windows += generate_windows((400, 496), (700, 1280), window_size=(96, 96), xy_step=(0.2, 0.2))
    windows += generate_windows((432, 528), (700, 1280), window_size=(96, 96), xy_step=(0.2, 0.2))
    windows += generate_windows((400, 528), (700, 1280), window_size=(128, 128), xy_step=(0.2, 0.2))
    windows += generate_windows((432, 560), (700, 1280), window_size=(128, 128), xy_step=(0.2, 0.2))
    windows += generate_windows((400, 596), (700, 1280), window_size=(196, 196), xy_step=(0.2, 0.2))
    windows += generate_windows((464, 660), (700, 1280), window_size=(196, 196), xy_step=(0.2, 0.2))
    windows += generate_windows((464, 720), (700, 1280), window_size=(244, 244), xy_step=(0.3, 0.3))
    windows += generate_windows((464, 720), (700, 1280), window_size=(244, 244), xy_step=(0.3, 0.3))

    vehicle_windows = []
    for window in windows:
        window_img = cv2.resize(img[window[0]:window[1], window[2]:window[3]], (64, 64))
        window_features = get_features(window_img, orient, pix_per_cell, cell_per_block)
        pred = svc.predict([window_features])
        if pred == 1:
            vehicle_windows.append(window)

    return vehicle_windows


def draw_windows(img, vehicle_windows):
    for window in vehicle_windows:
        cv2.rectangle(img,(window[2], window[0]), (window[3], window[1]),(255,0, 0),5)


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0]:box[1], box[2]:box[3]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def find_centroid(heatmap, x, y, w, h):
    weight = heatmap[y:y+h, x:x+w]
    total = np.sum(weight)
    x_weight = weight.sum(axis=0)/total
    y_weight = weight.sum(axis=1)/total
    x_seq = np.arange(x, x+w, 1)
    y_seq = np.arange(y, y+h, 1)
    cx = int(np.dot(x_weight, x_seq))
    cy = int(np.dot(y_weight, y_seq))
    return (cx, cy)


def find_closest_car_in_the_last_frame(org_img, vehicle_box, last_frame_vehicels):
    vx = vehicle_box[0]
    vy = vehicle_box[1]
    vw = vehicle_box[2]
    vh = vehicle_box[3]

    def find_overlap(pbox):
        (px, py, pw, ph) = pbox
        overlap_map = np.zeros_like(org_img)
        overlap_map[py:py + ph, px:px + pw] += 1
        overlap_map[vy:vy + vh, vx:vx + vw] += 1
        overlap_map[overlap_map < 2] = 0
        overlapping_area = np.sum(overlap_map) / 2
        return overlapping_area

    return max(last_frame_vehicels, key=find_overlap)


def find_vehicle_boxes(heatmap, org_img):
    global last_frame_vehicles
    global current_frame_vehicles

    last_frame_vehicles = current_frame_vehicles
    current_frame_vehicles = []

    heatmap2 = np.copy(heatmap)
    im2, contours, hierarchy = cv2.findContours(heatmap2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 30 or h < 30:
            continue
        if h / w > 1.5:
            continue
        centroid = find_centroid(heatmap, x, y, w, h)
        if last_frame_vehicles:
            prebox = find_closest_car_in_the_last_frame(org_img, (x, y, w, h), last_frame_vehicles)
            if prebox:
                if abs(float(w) - float(prebox[2])) / float(prebox[2]) < 0.5:
                    w = int((w + prebox[2]) / 2)

                if abs(float(h) - float(prebox[3])) / float(prebox[3]) < 0.5:
                    h = int((h + prebox[3]) / 2)

                x = int(centroid[0] - w / 2)
                y = int(centroid[1] - h / 2)

        current_frame_vehicles.append((x, y, w, h))
        cv2.rectangle(org_img, (x, y), (x + w, y + h), (0, 0, 255), 6)
    return org_img


def find_vehicles(org_img, v_windows):
    heatmap_img = np.zeros_like(org_img[:,:,0])
    heatmap_img = add_heat(heatmap_img, v_windows)
    heatmap_img = apply_threshold(heatmap_img, 2)
    img = org_img.copy()
    return find_vehicle_boxes(heatmap_img, img)


def process_image(image, clf, orient, pix_per_cell, cell_per_block):
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    v_windows = search_windows(img, clf, orient, pix_per_cell, cell_per_block)
    return find_vehicles(image, v_windows)

if __name__ == "__main__":
    work_dir = '/Users/wuyang/Downloads/self_driving_car/CarND-Vehicle-Detection/'
    orient = 8
    pix_per_cell = 16
    cell_per_block = 2

    svc_filename = 'svc_model.sav'
    svc = pickle.load(open(svc_filename, 'rb'))

    process_frame = lambda img: process_image(img, svc, orient=orient,
                                               pix_per_cell=pix_per_cell,
                                               cell_per_block=cell_per_block)

    last_frame_vehicles = []
    current_frame_vehicles = []

    output_v = 'project_video_output.mp4'
    clip1 = VideoFileClip(work_dir+"project_video.mp4")
    clip = clip1.fl_image(process_frame)
    clip.write_videofile(output_v, audio=False)