# -*- coding: utf-8 -*-

import os
import pickle
from collections import deque

import numpy as np
import moviepy.editor
import cv2
import matplotlib
import matplotlib.pyplot as plt

#import detect_object


CONFIG_KEYFRAME = {
    'CANNY_L_THRESHOLD': 25,
    'CANNY_H_THRESHOLD': 85
}

CONFIG_NO_KEYFRAME = {
}

CONFIG_LANE_DETECTION = {
    'PERSPECTIVE_MARGIN_REL': 0.40,
    'WINDOW_NUM': 55,
    'WINDOW_WIDTH_REL': 0.18,
    'LANE_THRESHOLD': 0.08,
    'LANE_MARGIN_SIGMA': 1.0
}

CONFIG_DETECT_CAR = {
    'THRESHOLD': 0.23,
    'CAR_INDEX': 6
}


def maybe_do_camera_calibration(folder_name, file_name):
    # For now, it supposes that the calibration has already finished.
    return


def obtain_camera_calibration_matrix(folder_name, file_name='camera_cal.p'):
    # The calibration process will save its result in the specified folder
    maybe_do_camera_calibration(folder_name, file_name)
    # Load the file
    dist_pickle = pickle.load(open(os.path.join(folder_name, file_name), "rb"))
    return dist_pickle["mtx"], dist_pickle["dist"]


def maybe_do_perspective_calibration(adj_img_filename):
    global CONFIG_LANE_DETECTION

    # Always do the setting currently
    adj_img = cv2.imread(adj_img_filename)
    top, bottom = 0, adj_img.shape[0]
    margin = CONFIG_LANE_DETECTION.get('PERSPECTIVE_MARGIN_REL')
    left = np.int(adj_img.shape[1] * margin)
    right = adj_img.shape[1] - left
    src, dst = [], []

    # Show the window
    def handler(e):
        if len(src) < 4:
            plt.axhline(int(e.ydata), linewidth=2, color='r')
            plt.axvline(int(e.xdata), linewidth=2, color='r')
            src.append((int(e.xdata), int(e.ydata)))
        if len(src) == 4:
            dst.extend([(left, bottom), (left, top), (right, top), (right, bottom)])
    was_interactive = matplotlib.is_interactive()
    if not was_interactive:
        plt.ion()
    fig = plt.figure()
    plt.imshow(adj_img)
    plt.axhline(470, linewidth=1, color='b')
    fig.canvas.mpl_connect('button_press_event', handler)
    fig.canvas.mpl_connect('close_event', lambda e: e.canvas.stop_event_loop())
    fig.canvas.start_event_loop(timeout=-1)

    # After the window is closed (by a user,) calculate the matrix
    mat_forward = cv2.getPerspectiveTransform(np.asfarray(src, np.float32), np.asfarray(dst, np.float32))
    mat_inverse = cv2.getPerspectiveTransform(np.asfarray(dst, np.float32), np.asfarray(src, np.float32))
    matplotlib.interactive(was_interactive)

    return mat_forward, mat_inverse


def obtain_perspective_calibration_matrix(adj_img_filename):
    # Show a window to specify the white lines
    mat_forward, mat_inverse = maybe_do_perspective_calibration(adj_img_filename)
    return mat_forward, mat_inverse


def threshold_and_binarize(img, is_key_frame):
    global CONFIG_KEYFRAME, CONFIG_NO_KEYFRAME
    config = CONFIG_KEYFRAME if is_key_frame else CONFIG_NO_KEYFRAME
    # Convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if is_key_frame:
        gray = cv2.Canny(gray, config.get('CANNY_L_THRESHOLD'), config.get('CANNY_H_THRESHOLD'))
    # Create binary based on detected pixels
    binary_threshold = np.zeros_like(gray)
    binary_threshold[(gray > 0)] = 1
    return binary_threshold


def get_normal_distribution(length, sigma):
    # TODO: The result of this should be shared in terms of calculation speed
    sigma3 = sigma * 3
    sigma_2p2 = 2 * (sigma ** 2)
    half_length = np.int(length * 0.5)
    scale = float(sigma3) * 2. / float(length)
    result = np.zeros(length, dtype=np.float)
    for i in range(length):
        x = float(i - half_length) * scale
        result[i] = np.exp((-((x**2)/2)) / sigma_2p2) / np.sqrt(sigma_2p2 * np.pi)
    return result


def draw_debug_image_on_lane_detection(warped_img, x, width, y_low, y_high, best_x, thresholded):
    cv2.rectangle(warped_img, (x, y_low), (x + width, y_high), (255, 255, 0), 2)
    col = (255, 0, 0) if thresholded else (0, 0, 255)
    cv2.circle(warped_img, (best_x+x, np.int((y_high+y_low)/2)), 5, col, 2)


def get_lines_with_sliding_window(warped_bin_img, debug_mode=False):
    global CONFIG_LANE_DETECTION

    # Get constants
    sigma = CONFIG_LANE_DETECTION.get('LANE_MARGIN_SIGMA')
    threshold = CONFIG_LANE_DETECTION.get('LANE_THRESHOLD')
    window_num = CONFIG_LANE_DETECTION.get('WINDOW_NUM')
    window_width = np.int(CONFIG_LANE_DETECTION.get('WINDOW_WIDTH_REL') * warped_bin_img.shape[1])
    window_height = np.int(warped_bin_img.shape[0] / window_num)
    default_l_x = np.int(warped_bin_img.shape[1] * CONFIG_LANE_DETECTION.get('PERSPECTIVE_MARGIN_REL'))
    default_r_x = warped_bin_img.shape[1] - default_l_x
    dist = get_normal_distribution(window_width, sigma)
    left_slope = np.linspace(0.5, 1.0, num=window_width)
    right_slope = np.linspace(1.0, 0.5, num=window_width)

    # Prepare result buffers
    left_x = [default_l_x]
    right_x = [default_r_x]
    left_y = [warped_bin_img.shape[0]]
    right_y = [warped_bin_img.shape[0]]

    # Prepare debug image if debug mode
    debug_img = None
    if debug_mode:
        debug_img = np.reshape(
            np.repeat(warped_bin_img, 3),
            newshape=(warped_bin_img.shape[0], warped_bin_img.shape[1], 3)
        ) * 255

    # Decide each point of the lines
    prev_l_x, prev_r_x = default_l_x, default_r_x
    for current_row in range(window_num):
        # Generate histogram for this row
        y_low = warped_bin_img.shape[0] - (current_row + 1) * window_height
        y_high = warped_bin_img.shape[0] - current_row * window_height
        histograms = np.sum(warped_bin_img[y_low:y_high, :], axis=0)

        # Generate windows for both lines
        left_window_x = np.clip(prev_l_x - np.int(window_width / 2), 0, warped_bin_img.shape[1] - window_width)
        left_window = (histograms[left_window_x:(left_window_x + window_width)]) / window_height
        left_window = np.multiply(left_window, dist)
        left_window = np.multiply(left_window, left_slope)
        right_window_x = np.clip(prev_r_x + np.int(window_width / 2), window_width, warped_bin_img.shape[1]) - window_width
        right_window = (histograms[right_window_x:(right_window_x + window_width)]) / window_height
        right_window = np.multiply(right_window, dist)
        right_window = np.multiply(right_window, right_slope)

        # Find the most plausible points
        best_l_x = np.argmax(left_window)
        best_r_x = np.argmax(right_window)

        # Draw debug image when debug mode
        if debug_mode:
            draw_debug_image_on_lane_detection(debug_img, left_window_x, window_width, y_low, y_high,
                                               best_l_x, left_window[best_l_x] >= threshold)
            draw_debug_image_on_lane_detection(debug_img, right_window_x, window_width, y_low, y_high,
                                               best_r_x, right_window[best_r_x] >= threshold)

        # Detect wrong line
        # TODO: We should add more rules
        if best_l_x + left_window_x > best_r_x + right_window_x:
            continue

        # Add a point for the left line
        if left_window[best_l_x] >= threshold:
            left_x.append(best_l_x + left_window_x)
            left_y.append(y_low)
            prev_l_x = best_l_x + left_window_x
        else:
            left_x.append(np.int(window_width / 2) + left_window_x)
            left_y.append(y_low)

        # Add a point for the right line
        if right_window[best_r_x] >= threshold:
            right_x.append(best_r_x + right_window_x)
            right_y.append(y_low)
            prev_r_x = best_r_x + right_window_x
        else:
            left_x.append(np.int(window_width / 2) + right_window_x)
            left_y.append(y_low)

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    return left_fit, right_fit, debug_img


def draw_lane(undistorted_img, binary_warped, left_fit, right_fit, warp_mat_inv):
    global CONFIG_DETECT_CAR

    # Detect other cars
    #cars = detect_object.get_cars(
    #    undistorted_img, CONFIG_DETECT_CAR.get('THRESHOLD'), CONFIG_DETECT_CAR.get('CAR_INDEX'))

    # Create an image to draw the lines on
    warped_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warped = np.dstack((warped_zero, warped_zero, warped_zero))

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warped, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img_size = (undistorted_img.shape[1], undistorted_img.shape[0])
    unwarped_img = cv2.warpPerspective(color_warped, warp_mat_inv, img_size, flags=cv2.INTER_LINEAR)

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_img, 1, unwarped_img, 0.3, 0)

    #for (x, y, w, h) in cars:
    #    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return result


def process_an_image(
        original_img,
        camera_cal_mtx,
        camera_dist,
        warp_mtx,
        warp_mtx_inv,
        results_buf_left,
        results_buf_right,
        results_weight,
        current_frame_id,
        debug_mode=False):

    # Un-distort the image firstly
    undistorted_img = cv2.undistort(original_img, camera_cal_mtx, camera_dist, None, camera_cal_mtx)

    # We could consider key frames that may be processed more accurately but computationally expensive
    # e.g.) is_key_frame = (current_frame_id % 25 == 0)
    is_key_frame = True

    # Filter (For now it applies no filter)
    filtered_img = undistorted_img

    # Binarize with specific threshold
    threshold_bin_img = threshold_and_binarize(filtered_img, is_key_frame)

    # Warp
    img_size = (filtered_img.shape[1], filtered_img.shape[0])
    warped_bin_img = cv2.warpPerspective(threshold_bin_img, warp_mtx, img_size, flags=cv2.INTER_LINEAR)

    # Get the lines
    if debug_mode:
        left_line, right_line, dimg = get_lines_with_sliding_window(warped_bin_img, True)
        return dimg

    left_line, right_line, _ = get_lines_with_sliding_window(warped_bin_img)
    results_buf_left.append(left_line)
    results_buf_right.append(right_line)

    # Draw the lines
    final_img = draw_lane(
        undistorted_img,
        warped_bin_img,
        np.average(results_buf_left, 0, results_weight[-len(results_buf_left):]),    # Make it smooth
        np.average(results_buf_right, 0, results_weight[-len(results_buf_right):]),  # Make it smooth
        warp_mtx_inv)

    return final_img


def get_image_filter_for_every_frame(camera_cal_mtx, camera_dist, warp_mtx, warp_mtx_inv):
    # Buffer to store some previous results
    result_buf_size = 80
    results_buf_left, results_buf_right = deque(maxlen=result_buf_size), deque(maxlen=result_buf_size)
    results_weight = np.arange(1, result_buf_size + 1) / result_buf_size
    current_frame_id = [0]

    def image_filter_for_every_frame(frame_img):
        current_frame_id[0] = current_frame_id[0] + 1
        processed_img = process_an_image(
            frame_img,
            camera_cal_mtx,
            camera_dist,
            warp_mtx,
            warp_mtx_inv,
            results_buf_left,
            results_buf_right,
            results_weight,
            current_frame_id[0])
        return processed_img

    return image_filter_for_every_frame


def main_process(in_video_filename, out_video_filename, camera_config_folder, adj_img_filename,
                 dnn_model_pb_filename):
    # Prepare the DNN model for detecting vehicles
    #graph = detect_object.load_graph(dnn_model_pb_filename)
    #detect_object.prepare(graph)

    # Load camera calibration config first
    camera_cal_mtx, camera_dist = obtain_camera_calibration_matrix(camera_config_folder)

    # Get config for warping images according to the lines for the lane
    warp_mtx, warp_mtx_inv = obtain_perspective_calibration_matrix(adj_img_filename)

    # Process the movie
    print('Now start to process the video')
    input_clip = moviepy.editor.VideoFileClip(in_video_filename)
    handler = get_image_filter_for_every_frame(
        camera_cal_mtx, camera_dist, warp_mtx, warp_mtx_inv)
    moviepy_session = input_clip.fl_image(handler)
    moviepy_session.write_videofile(out_video_filename, audio=False)

    # Finish
    print('Done.')
    return


if __name__ == '__main__':
    # Default file names
    data_folder = os.path.join('..', 'lane-detection-data')
    arg_in_video_filename = os.path.join(data_folder, 'exampleB.mp4')
    arg_out_video_filename = os.path.join(data_folder, 'out', 'exampleB.mp4')
    arg_camera_config_folder = os.path.join(data_folder, 'camera_cal')
    arg_adj_img_filename = os.path.join(data_folder, 'exampleB.jpg')
    arg_dnn_model_pb_filename = os.path.join(data_folder, 'tiny-yolo-voc.pb')

    # Execute the annotation process
    main_process(
        arg_in_video_filename,
        arg_out_video_filename,
        arg_camera_config_folder,
        arg_adj_img_filename,
        arg_dnn_model_pb_filename)
