# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
import numpy as np
from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor

sess_global = {}
graph_global = {}


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def get_tf_config(use_gpu=False):
    cfg = dict({
        'allow_soft_placement': False,
        'log_device_placement': False
    })

    if use_gpu:
        cfg['gpu_options'] = tf.GPUOptions(
            per_process_gpu_memory_fraction=1., allow_growth=True)
        cfg['allow_soft_placement'] = True
    else:
        cfg['device_count'] = {'GPU': 0}

    return tf.ConfigProto(**cfg)


def resize_input(im, w, h):
    reseized_im = cv2.resize(im, (int(w), int(h)))
    reseized_im = reseized_im / 255.
    reseized_im = reseized_im[:, :, ::-1]
    return reseized_im


def prepare(graph):
    global sess_global
    global graph_global
    graph_global = graph
    sess_global = tf.Session(graph=graph_global, config=get_tf_config())


def find_boxes(net_out, threshold):
    boxes = box_constructor(
        {
            'thresh': threshold,
            'anchors': [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
            'out_size': [13, 13, 125],
            'classes': 20,
            'num': 5
        }, net_out)
    return boxes


def get_box_and_index_and_prob(b, threshold, w, h):
    max_indx = np.argmax(b.probs)
    max_prob = b.probs[max_indx]
    left = int((b.x - b.w/2.) * w)
    right = int((b.x + b.w/2.) * w)
    top = int((b.y - b.h/2.) * h)
    bot = int((b.y + b.h/2.) * h)
    if left  < 0    :  left = 0
    if right > w - 1: right = w - 1
    if top   < 0    :   top = 0
    if bot   > h - 1:   bot = h - 1
    return left, top, (right-left), (bot - top), max_indx, max_prob


def get_cars(im, threshold, car_index):
    global sess_global
    global graph_global
    orig_h, orig_w = im.shape[0], im.shape[1]
    new_h, new_w = 416., 416.
    new2orig_x, new2orig_y = orig_w / new_w, orig_h / new_h
    im = resize_input(im, new_w, new_h)
    im = np.expand_dims(im, 0)
    in_tensor = {graph_global.get_tensor_by_name('prefix/input:0'): im}
    out_tensor = sess_global.run(graph_global.get_tensor_by_name('prefix/output:0'), in_tensor)[0]
    cars_box = find_boxes(out_tensor, threshold)
    cars = []
    for c in cars_box:
        x, y, w, h, idx, prob = get_box_and_index_and_prob(c, threshold, new_w, new_h)
        if idx != car_index or prob < threshold:
            continue
        cars.append(
            (int(x * new2orig_x), int(y * new2orig_y),
             int(w * new2orig_x), int(h * new2orig_y))
        )
    return cars
