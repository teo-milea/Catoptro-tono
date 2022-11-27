# Import TF and TF Hub libraries.
from turtle import pos
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import requests
import imutils
import time
from multiprocessing.dummy import Pool
from yeelight import Bulb

URL = "http://192.168.1.145:8000"
#IP_green_light = "192.168.1.145"
#red_green_bulb = Bulb(IP_green_light)

# red_green_bulb.set_brightness(100)
# red_green_bulb.set_rgb(0,255,0)
# activity_bulb = Bulb("192.168.1.140")
# activity_bulb.set_brightness(100)
# activity_bulb.set_rgb(255,255,255)
# activity_bulb.start_music()
requests_pool = Pool(6)
processing_pool = Pool(6)
play_status = False
future_requests = []
requests_to_process = []
last_pos_found = None
bec_on = True
# history = []

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (255,0,255),
    (0, 2): (0,255,255),
    (1, 3): (255,0,255),
    (2, 4): (0,255,255),
    (0, 5): (255,0,255),
    (0, 6): (0,255,255),
    (5, 7): (255,0,255),
    (7, 9): (255,0,255),
    (6, 8): (0,255,255),
    (8, 10): (0,255,255),
    (5, 6): (255,255,0),
    (5, 11): (255,0,255),
    (6, 12): (0,255,255),
    (11, 12): (255,255,0),
    (11, 13): (255,0,255),
    (13, 15): (255,0,255),
    (12, 14): (0,255,255),
    (14, 16): (0,255,255)
}

times = []
left_arm_state = 0
right_arm_state = 0

def derivative_first(x, t):
    return (x[2] - x[0]) / (t[2] - t[0])

def derivative_second(x, t):
    return (x[2] - 2 * x[1] + x[0]) * 4 / ((t[2] - t[0]) ** 2)

def position(history, threshold):    
    keypoints = history[-1]
    left_shoulder = keypoints[0,0,KEYPOINT_DICT['left_shoulder'],:].numpy()
    right_shoulder = keypoints[0,0,KEYPOINT_DICT['right_shoulder'],:].numpy()
    left_hip = keypoints[0,0,KEYPOINT_DICT['left_hip'],:].numpy()
    right_hip = keypoints[0,0,KEYPOINT_DICT['right_hip'],:].numpy()

    f = lambda x: sum(x)/4
    center_points = [left_shoulder, right_shoulder, left_hip, right_hip] 
    
    for c in center_points:
        if c[2] < threshold:
            return None
    
    x = f([x[1] for x in center_points])
    y = f([x[0] for x in center_points])
    
    return x,y

def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

def get_depth(history, threshold, pos):
    # pos = position(history, threshold)
    if pos:
        x_center, y_center = pos
        keypoints = history[-1]
        nose = keypoints[0,0,KEYPOINT_DICT['nose'],:].numpy()
    
        if nose[2] > threshold:
            return distanceCalculate((x_center,y_center),(nose[1], nose[0]))
    return None

def are_hips_missing(history, threshold):
    keypoints = history[-1]
    left_hip = keypoints[0,0,KEYPOINT_DICT['left_hip'],:].numpy()
    nose = keypoints[0,0,KEYPOINT_DICT['nose'],:].numpy()
    right_hip = keypoints[0,0,KEYPOINT_DICT['right_hip'],:].numpy()
    if left_hip[2] < threshold and right_hip[2] < threshold and nose[2] > threshold:
        return True
    return None

def find_left_arm(history, threshold):
    keypoints = history[-1]

    left_wrist = keypoints[0,0,KEYPOINT_DICT['left_elbow'],:].numpy()
    x_left_arm = left_wrist[1]
    y_left_arm = left_wrist[0]

    # pos = position(history, threshold)
    left_shoulder = keypoints[0,0,KEYPOINT_DICT['left_shoulder'],:].numpy()
    
    if left_shoulder[2] > threshold:
        x_center = left_shoulder[1]
        y_center = left_shoulder[0]

        if left_wrist[2] > threshold:
            distance = distanceCalculate((x_center,y_center),(x_left_arm, y_left_arm))
            sin_left_arm = (y_left_arm - y_center) / distance
            return distance, sin_left_arm
    return None

def find_right_arm(history, threshold):
    keypoints = history[-1]

    right_wrist = keypoints[0,0,KEYPOINT_DICT['right_elbow'],:].numpy()
    x_right_arm = right_wrist[1]
    y_right_arm = right_wrist[0]

    # pos = position(history, threshold)
    right_shoulder = keypoints[0,0,KEYPOINT_DICT['right_shoulder'],:].numpy()
    
    if right_shoulder[2] > threshold:
        x_center = right_shoulder[1]
        y_center = right_shoulder[0] 

        if right_wrist[2] > threshold:
            distance = distanceCalculate((x_center,y_center),(x_right_arm, y_right_arm))
            sin_right_arm = (y_right_arm - y_center) / distance
            return distance, sin_right_arm
    return None

#def change_light(r,g,b):
#    global red_green_bulb
#    if not bec_on:
#        return
#    try:
#        red_green_bulb.set_rgb(r,g,b)
#   except:
#        red_green_bulb = Bulb(IP_green_light)
#        red_green_bulb.set_rgb(r,g,b)

def activity_level(history, threshold, all_points_derivatives):

    if len(history) < 3:
        return None

    found_points = 0
    first_dev = 0
    second_dev = 0
    for i in [7,8]:

        res = all_points_derivatives[-1][i]
        if res != None :
                found_points += 1
                local_first_dev, local_second_dev = res
                first_dev += local_first_dev
                second_dev += local_second_dev

    if found_points > 0:
        return first_dev / found_points, second_dev / found_points

    return None

def new_look_front(history, threshold):
    keypoints = history[-1]
    nose = keypoints[0,0,KEYPOINT_DICT['nose'],:].numpy()
    left_ear = keypoints[0,0,KEYPOINT_DICT['left_ear'],:].numpy()
    right_ear = keypoints[0,0,KEYPOINT_DICT['right_ear'],:].numpy()

    if nose[2] < threshold:
        return None

    if left_ear[2] > threshold and right_ear[2] > threshold:
        return nose[1] < left_ear[1] and nose[1] > right_ear[1]
    else:
        left_eye = keypoints[0,0,KEYPOINT_DICT['left_eye'],:].numpy()
        right_eye = keypoints[0,0,KEYPOINT_DICT['right_eye'],:].numpy()
    
        if left_eye[2] < threshold and right_eye[2] < threshold:
            return nose[1] < left_eye[1] and nose[1] > right_eye[1]
    return None

def point_acc(history, threshold, point_index):
    global times

    if len(history) < 3:
        return None

    right_arm_2 = history[-1][0,0,point_index,:].numpy()
    right_arm_1  = history[-2][0,0,point_index,:].numpy()
    right_arm_0  = history[-3][0,0,point_index,:].numpy()

    first_dev = 0
    second_dev = 0

    if right_arm_0[2] > threshold and right_arm_1[2] > threshold and right_arm_2[2] > threshold:
        selected_times = times[-3:]
        first_dev += abs(derivative_first([right_arm_0[0], right_arm_1[0], right_arm_2[0]], selected_times))
        first_dev += abs(derivative_first([right_arm_0[1], right_arm_1[1], right_arm_2[1]], selected_times))
        second_dev += abs(derivative_second([right_arm_0[0], right_arm_1[0], right_arm_2[0]], selected_times))
        second_dev += abs(derivative_second([right_arm_0[1], right_arm_1[1], right_arm_2[1]], selected_times))
        return first_dev, second_dev
    else:
        return None

def trigger_arm_acc(history, threshold, arm_side="left"):
    global times

    if len(history) < 3:
        return None

    if arm_side == "left":
        index_wrist = KEYPOINT_DICT["right_elbow"]
    elif arm_side == "right":
        index_wrist = KEYPOINT_DICT["left_elbow"]
    else:
        raise Exception("wrong arm index; use left or right")

    wrist_2 = history[-1][0,0,index_wrist,2]
    wrist_1 = history[-2][0,0,index_wrist,2]
    wrist_0 = history[-3][0,0,index_wrist,2]
    
    if wrist_0 < threshold or wrist_1 < threshold or wrist_2 < threshold:
        return None

    # acc = derivative_second([wrist_0[0], wrist_1[0], wrist_2[0]], times[-3:])
    acc = all_points_derivatives[-1][index_wrist][1]
    return acc

def draw_viz(img, keypoints, x, y, threshold):
    # iterate through keypoints
    for k in keypoints[0,0,:,:]:
        # Converts to numpy array
        k = k.numpy()

        # Checks confidence for keypoint
        if k[2] > threshold:
            # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
            yc = int(k[0] * y)
            xc = int(k[1] * x)

            # Draws a circle on the image for each keypoint
            img = cv2.circle(img, (xc, yc), 2, (0, 255, 0), 5)
                
    def convert(value):
        return (int(value[1] * x), int(value[0] * y), value[2])
    
    for (u,v),c  in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        x_start, y_start, conf_start = convert(keypoints[0,0,u,:].numpy())
        x_end, y_end, conf_end = convert(keypoints[0,0,v,:].numpy())
        if conf_start > threshold and conf_end > threshold:
            img = cv2.line(img, (x_start, y_start), (x_end, y_end), c)
    
    return img

def pos_and_depth_fn(history, threshold, allow_request_sending=True):
    global future_requests, requests_to_process, last_pos_found
    pos = position(history, threshold)

    last_pos = None

    if pos:
        x_pos, y_pos = pos
        last_pos_found = pos

        payload = {"pan": -(x_pos * 2 - 1)}
        last_pos = -(x_pos * 2 - 1)

        if allow_request_sending:        
            future_r = requests_pool.apply_async(
                        requests.post, 
                        args=[URL + '/set_pan', ], 
                        kwds={"params": payload, "timeout": 0.1}
                    )
            requests_to_process.append(future_r)

    depth = get_depth(history, threshold, pos)
    if depth != None:
        payload = {"volume": min(max((depth - 0.1) * 2.5, 0),1)}
        if allow_request_sending:
            future_r = requests_pool.apply_async(
                        requests.post, 
                        args=[URL + '/set_volume', ], 
                        kwds={"params": payload, "timeout": 0.1}
                    )
            requests_to_process.append(future_r)

def left_arm_fn(history, threshold, allow_request_sending=True):
    global requests_to_process
    res = find_left_arm(history, threshold)
    if res:
        dis_left, sin_left = res
        payload = {"distance": dis_left * 2, "angle": sin_left}
        if allow_request_sending:
            future_r = requests_pool.apply_async(
                        requests.post, 
                        args=[URL + '/left_arm', ], 
                        kwds={"params": payload, "timeout": 0.1}
                    )
            requests_to_process.append(future_r)
    

def right_arm_fn(history, threshold, allow_request_sending=True):
    global requests_to_process
    res = find_right_arm(history, threshold)
    if res:
        dis_right, sin_right = res
        payload = {"distance": dis_right * 2, "angle": sin_right}
        if allow_request_sending:
            future_r = requests_pool.apply_async(
                        requests.post, 
                        args=[URL + '/right_arm', ], 
                        kwds={"params": payload, "timeout": 0.1}
                    )
            requests_to_process.append(future_r)

def activity_level_fn(history, threshold, all_points_derivatives, allow_request_sending=True):
    global requests_to_process
    res = activity_level(history, threshold, all_points_derivatives)
    if res:
        first_order, second_order = res
        x1 = ((first_order - 15/256) + 1) / 2
        payload = {
            "first_order": (1.45 - 0.32/(0.2 + x1/2)) * 128, 
            "second_order": ((second_order - 15/12.8) + 5) * 12.8 * 4
            }
        if allow_request_sending:
            future_r = requests_pool.apply_async(
                        requests.post, 
                        args=[URL + '/activity_level', ], 
                        kwds={"params": payload, "timeout": 0.1}
                    )
            requests_to_process.append(future_r)
        # activity_bulb.set_brightness((first_order + second_order) / 2 / 128 * 100)

def look_front_fn(history, threshold, allow_request_sending=True):
    global requests_to_process
    # global current_look_front
    res = new_look_front(history, threshold)
    current_look_front = False
    if res is not None:
        payload = {"look_front": res}
        if allow_request_sending:
            future_r = requests_pool.apply_async(
                        requests.post, 
                        args=[URL + '/look_front', ], 
                        kwds={"params": payload, "timeout": 0.1}
                    )
            requests_to_process.append(future_r)
        current_look_front = res
    else:
        payload = {"look_front": current_look_front}
        if allow_request_sending:
            future_r = requests_pool.apply_async(
                        requests.post, 
                        args=[URL + '/look_front', ], 
                        kwds={"params": payload, "timeout": 0.1}
                    )
            requests_to_process.append(future_r)
            

def trigger_left_arm_acc_fn(all_points_derivatives, allow_request_sending=True):
    global requests_to_process
    res = all_points_derivatives[-1][KEYPOINT_DICT["left_elbow"]]
    if res != None:
        payload = {"speed": res[0],"acc": res[1]}
        if allow_request_sending:
            future_r = requests_pool.apply_async(
                        requests.post, 
                        args=[URL + '/trigger_left_arm_acc', ], 
                        kwds={"params": payload, "timeout": 0.1}
                    )
            requests_to_process.append(future_r)        

def trigger_right_arm_acc_fn(all_points_derivatives, allow_request_sending=True):
    global requests_to_process
    res = all_points_derivatives[-1][KEYPOINT_DICT["right_elbow"]]
    if res != None:
        payload = {"speed": res[0],"acc": res[1]}
        if allow_request_sending:
            future_r = requests_pool.apply_async(
                        requests.post, 
                        args=[URL + '/trigger_right_arm_acc', ], 
                        kwds={"params": payload, "timeout": 0.1}
                    )
            requests_to_process.append(future_r)

def is_person_in_the_room_fn(history, threshold, allow_request_sending=True):
    global play_status, requests_to_process
    res = is_person_in_the_room(history, threshold)
    if res != None:
        if res == True and play_status == False:
            if last_pos_found[1] < 0.25:
                if allow_request_sending:
                    future_r = requests_pool.apply_async(
                                requests.post, 
                                args=[URL + '/start_play', ], 
                                kwds={"timeout": 0.1}
                            )
                    requests_to_process.append(future_r)
    #                red_green_bulb.set_rgb(255,0,0)
                change_light(255,0,0)
            else:
                if allow_request_sending:
                    future_r = requests_pool.apply_async(
                                requests.post, 
                                args=[URL + '/unpause_play', ], 
                                kwds={"timeout": 0.1}
                            )
                    requests_to_process.append(future_r)
    #                red_green_bulb.set_rgb(255,0,0)
                # change_light(255,0,0)

            play_status = True
            
        elif res == False and play_status == True:
            if allow_request_sending:
                future_r = requests_pool.apply_async(
                            requests.post, 
                            args=[URL + '/end_play', ], 
                            kwds={"timeout": 0.1}
                        )
                requests_to_process.append(future_r)
    #                red_green_bulb.set_rgb(0,255,0)
                if last_pos_found[1] < 0.25:
                    change_light(0,255,0)
                play_status = False

def pretty_print_dict(d):
    print()
    last_value = 0
    total_sum = 0 
    for k, v in d:
        time_value = v - last_value
        total_sum += time_value
        last_value = v
        print(k, time_value, last_value)
    print("total:", total_sum)
    print()

def is_person_in_the_room(history, threshold):
    global play_status
    if len(history) < 3:
        return None
    values = history[-3:]

    points_needed_true = 7
    points_needed_false = 4
    points_found = 0
    for h in values:
        for i in range(17):
            if h[0,0,i,2] > threshold:
                points_found +=1

    if play_status == False:
        res = points_found/3 > points_needed_true
        print(f"PLAY_STATUS: {play_status}// RES: {res}")
        # return 
        if res == True:
            return True
    else:
        res = points_found/3 < points_needed_false
        print(f"PLAY_STATUS: {play_status}// RES: {res}")
        if res == True:
            return False
    print("UNSURE")
    return None

def main():
    # Download the model from TF Hub.
    # model = hub.load('https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/4?lite-format=tflite')

    model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/4')
    movenet = model.signatures['serving_default']

    # Threshold for 
    threshold = 0.33

    # Loads video source (0 is for main webcam)
    video_source = 0
    # vid_capture = cv2.VideoCapture('/home/teo/Documents/GOPR0003.MP4')
    cap = cv2.VideoCapture(video_source)
    
    global times, requests_to_process
    success, img = cap.read()
    times.append(time.time())

   # change_light(0,255,0)

    if not success:
       print('Error reding frame')
       quit()

    history = []
    all_points_derivatives = []
    last_look_front = None

    # requests_to_process = []

    total_fps = 0
    total_index = 0
    
    while success:
        # Reads next frame

        y, x, _ = img.shape
    # while True:
        # A frame of video or an image, represented as an int32 tensor of shape: 256x256x3. Channels order: RGB with values in [0, 255].
        tf_img = cv2.resize(img, (256,256))
        tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
        tf_img = np.asarray(tf_img)
        tf_img = np.expand_dims(tf_img,axis=0)

        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image = tf.cast(tf_img, dtype=tf.int32)

        # Run model inference.
        outputs = movenet(image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints = outputs['output_0']
        history.append(keypoints)
                
        img = draw_viz(img, keypoints, x, y, threshold) 
        
        if len(times) > 2:
            current = 1/(times[-1] - times[-2])
            print("FPS:", current)
            total_fps += current
            total_index += 1
            print("AVG FPS:",total_fps/total_index)
            if current < 1:
                return

        allow_request_sending = True
        allow_gesture_recognition = True
        sync_mode = False

        start_processing_time = time.time()
        if allow_gesture_recognition:
            time_records = []
            
            points_derivates = {}
            for i in [7, 8]:
                point_der = point_acc(history, threshold, i)
                points_derivates[i] = point_der
            all_points_derivatives.append(points_derivates)
            derivatives_time = time.time() - start_processing_time
            time_records.append(("derivate", derivatives_time))

            if sync_mode == True:
                pos = position(history, threshold)

                last_pos = None

                if pos:
                    x_pos, y_pos = pos
                    xc = int(x_pos * x)
                    yc = int(y_pos * y)
                    
                    img = cv2.circle(img, (xc, yc), 2, (0, 255, 0), 5)
                    
                    payload = {"pan": -(x_pos * 2 - 1)}
                    last_pos = -(x_pos * 2 - 1)

                    if allow_request_sending:        
                        future_r = requests_pool.apply_async(
                                    requests.post, 
                                    args=[URL + '/set_pan', ], 
                                    kwds={"params": payload, "timeout": 0.1}
                                )
                        requests_to_process.append(future_r)

                position_time = time.time() - start_processing_time
                time_records.append(("position", position_time))

                depth = get_depth(history, threshold, pos)
                if depth != None:
                    payload = {"volume": min(max((depth - 0.1) * 2.5, 0),1)}
                    if allow_request_sending:
                        future_r = requests_pool.apply_async(
                                    requests.post, 
                                    args=[URL + '/set_volume', ], 
                                    kwds={"params": payload, "timeout": 0.1}
                                )
                        requests_to_process.append(future_r)

                depth_time = time.time() - start_processing_time
                time_records.append(("depth", depth_time))

                res = find_left_arm(history, threshold)
                if res:
                    dis_left, sin_left = res
                    payload = {"distance": dis_left * 2, "angle": sin_left}
                    if allow_request_sending:
                        future_r = requests_pool.apply_async(
                                    requests.post, 
                                    args=[URL + '/left_arm', ], 
                                    kwds={"params": payload, "timeout": 0.1}
                                )
                        requests_to_process.append(future_r)

                left_arm_time = time.time() - start_processing_time
                time_records.append(("left_arm", left_arm_time))

                res = find_right_arm(history, threshold)
                if res:
                    dis_right, sin_right = res
                    payload = {"distance": dis_right * 2, "angle": sin_right}
                    if allow_request_sending:
                        future_r = requests_pool.apply_async(
                                    requests.post, 
                                    args=[URL + '/right_arm', ], 
                                    kwds={"params": payload, "timeout": 0.1}
                                )
                        requests_to_process.append(future_r)
                right_arm_time = time.time() - start_processing_time
                time_records.append(("right_arm", right_arm_time))

                res = activity_level(history, threshold, all_points_derivatives)
                if res:
                    first_order, second_order = res
                    x1 = ((first_order - 15/256) + 1) / 2
                    payload = {
                        "first_order": (1.45 - 0.32/(0.2 + x1/2)) * 128, 
                        "second_order": ((second_order - 15/12.8) + 5) * 12.8 * 4
                        }
                    if allow_request_sending:
                        future_r = requests_pool.apply_async(
                                    requests.post, 
                                    args=[URL + '/activity_level', ], 
                                    kwds={"params": payload, "timeout": 0.1}
                                )
                        requests_to_process.append(future_r)
                activity_level_time = time.time() - start_processing_time
                time_records.append(("activity_level", activity_level_time))

                current_look_front = False
                res = new_look_front(history, threshold)
                if res is not None:
                    payload = {"look_front": res}
                    if allow_request_sending:
                        future_r = requests_pool.apply_async(
                                    requests.post, 
                                    args=[URL + '/look_front', ], 
                                    kwds={"params": payload, "timeout": 0.1}
                                )
                        requests_to_process.append(future_r)
                    current_look_front = res
                else:
                    payload = {"look_front": current_look_front}
                    if allow_request_sending:
                        future_r = requests_pool.apply_async(
                                    requests.post, 
                                    args=[URL + '/look_front', ], 
                                    kwds={"params": payload, "timeout": 0.1}
                                )
                        requests_to_process.append(future_r)
                look_front_time = time.time() - start_processing_time
                time_records.append(("look_front", look_front_time))

                res = all_points_derivatives[-1][KEYPOINT_DICT["left_wrist"]]
                if res != None:
                    payload = {"acc": res[1]}
                    if allow_request_sending:
                        future_r = requests_pool.apply_async(
                                    requests.post, 
                                    args=[URL + '/trigger_left_arm_acc', ], 
                                    kwds={"params": payload, "timeout": 0.1}
                                )
                        requests_to_process.append(future_r)
                trigger_left_arm_acc_time = time.time() - start_processing_time
                time_records.append(("trigger_left_arm_acc", trigger_left_arm_acc_time))

                res = all_points_derivatives[-1][KEYPOINT_DICT["right_wrist"]]
                if res != None:
                    payload = {"acc": res[1]}
                    if allow_request_sending:
                        future_r = requests_pool.apply_async(
                                    requests.post, 
                                    args=[URL + '/trigger_right_arm_acc', ], 
                                    kwds={"params": payload, "timeout": 0.1}
                                )
                        requests_to_process.append(future_r)
                trigger_right_arm_acc_time = time.time() - start_processing_time
                time_records.append(("trigger_right_arm_acc", trigger_right_arm_acc_time))

                pretty_print_dict(time_records)

            else:
                processes = []
                future = processing_pool.apply_async(
                    left_arm_fn,
                    args=[history, threshold],
                    kwds={"allow_request_sending": allow_request_sending}
                )
                processes.append(future)
                future = processing_pool.apply_async(
                    right_arm_fn,
                    args=[history, threshold],
                    kwds={"allow_request_sending": allow_request_sending}
                )
                processes.append(future)
                future = processing_pool.apply_async(
                    look_front_fn,
                    args=[history, threshold],
                    kwds={"allow_request_sending": allow_request_sending}
                )
                processes.append(future)
                future = processing_pool.apply_async(
                    pos_and_depth_fn,
                    args=[history, threshold],
                    kwds={"allow_request_sending": allow_request_sending}
                )
                processes.append(future)
                future = processing_pool.apply_async(
                    activity_level_fn,
                    args=[history, threshold, all_points_derivatives],
                    kwds={"allow_request_sending": allow_request_sending}
                )
                processes.append(future)
                future = processing_pool.apply_async(
                    trigger_left_arm_acc_fn,
                    args=[all_points_derivatives],
                    kwds={"allow_request_sending": allow_request_sending}
                )
                processes.append(future)
                future = processing_pool.apply_async(
                    trigger_right_arm_acc_fn,
                    args=[all_points_derivatives],
                    kwds={"allow_request_sending": allow_request_sending}
                )
                processes.append(future)
                future = processing_pool.apply_async(
                    is_person_in_the_room_fn,
                    args=[history, threshold],
                    kwds={"allow_request_sending": allow_request_sending}
                )
                processes.append(future)
                for f in processes:
                    f.get()

                for f in requests_to_process:
                   try:
                       f.get()
                   except:
                       pass          

        # Shows image
        cv2.imshow('Movenet', img)
        success, img = cap.read()
        times.append(time.time())

        # Waits for the next frame, checks if q was pressed to quit
        if cv2.waitKey(1) == ord("q"):
           break


    cap.release()


if __name__ == "__main__":
    main()
