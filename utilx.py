import copy
import cv2
import simpleaudio as sa
import numpy as np

CENTROID_HISTORY_LENGTH = 10
AUDIO_FILES = ["505/kick.wav", "505/hh.wav", "505/hhp.wav",
               "505/snare.wav", "505/kick2.wav", "505/snare2.wav"]
wave_obj1 = sa.WaveObject.from_wave_file(AUDIO_FILES[1])
wave_obj2 = sa.WaveObject.from_wave_file(AUDIO_FILES[3])
wave_obj3 = sa.WaveObject.from_wave_file(AUDIO_FILES[2])
wave_obj4 = sa.WaveObject.from_wave_file(AUDIO_FILES[0])
wave_obj6 = sa.WaveObject.from_wave_file(AUDIO_FILES[4])
wave_obj5 = sa.WaveObject.from_wave_file(AUDIO_FILES[5])
WAVE_OBJECTS = [wave_obj1, wave_obj2, wave_obj3,
                wave_obj5, wave_obj6, wave_obj4]
drums = ["hat", "snare", "hhp", "kick", "kick2", "snare2"]


def find_largest_contour(contours):
    largest_index = 0
    max_area = 0
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if (area > max_area):
            max_area = area
            largest_index = i
    return contours[largest_index]


def find_largest_x_contours(contours, k):
    largest_n = copy.deepcopy(contours)
    n = len(largest_n)
    for i in range(n):
        min_index = i
        min_value = cv2.contourArea(largest_n[i])
        for j in range(i+1, n):
            t_area = cv2.contourArea(largest_n[j])
            if t_area < min_value:
                min_index = j
                min_value = t_area
                largest_n[i], largest_n[min_index] \
                    = largest_n[min_index], largest_n[i]

    # for l in largest_n:
    #     print(cv2.contourArea(l))
    return largest_n[-k:]


RECTS = {
    "01": [100, 10, 300-100, 110-10],
    "02": [300, 10, 500-300, 110-10],
    "03": [10, 110, 120-10, 380-110],
    "04": [10, 400, 120-10, 580-400],
    "05": [480, 110, 580-480, 380-110],
    "06": [480, 400, 580-480, 580-400],
}

toggles = {
    "01": False,
    "02": False,
    "03": False,
    "04": False,
    "05": False,
    "06": False,
}

REGION_INDEX = ["01", "02", "03", "04", "05", "06"]
hitted = {
    "01": 0,
    "02": 0,
    "03": 0,
    "04": 0,
    "05": 0,
    "06": 0,
}


def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return ()
    return (x, y, w, h)


def if_overlap(canvas, hand_rect=[0, 0, 0, 0], speed=1000):
    old_rects = toggles.copy()
    rects = list(RECTS.values())
    points = hand_rect
    for i, r in enumerate(rects):
        grouped = intersection(points, r)
        if len(grouped) > 0:
            toggles[f"0{i+1}"] = True
        if len(grouped) == 0:
            toggles[f"0{i+1}"] = False

    if speed < 70:
        return 0
    for t in REGION_INDEX:
        if old_rects[t] != toggles[t] and not old_rects[t]:
            playsound(canvas, t)
    return 0


def playsound(canvas, region_index):
    points = RECTS[region_index]
    pt1 = (int(points[0]), int(points[1]))
    pt2 = (int(points[2] + points[0]), int(points[3]+points[1]))
    canvas = cv2.rectangle(canvas, pt1, pt2,
                           (0, 255, 0), thickness=-1)
    r = int(region_index[-1])
    WAVE_OBJECTS[r-1].play()


def draw_rectangles(canvas):
    box_color = (230, 220, 230)
    fill_color = (130, 120, 130)
    line_width = 4
    fill_width = -1

    for i in range(6):
        tmp = f"0{i+1}"
        points = RECTS[tmp]

        drumname = f"({drums[i]})"
        if toggles[tmp] is False:
            clr = box_color
            fill = 1
        else:
            clr = fill_color
            fill = -1
        pt1 = (int(points[0]), int(points[1]))
        pt2 = (int(points[2] + points[0]), int(points[3]+points[1]))
        canvas = cv2.rectangle(canvas, pt1, pt2,
                               clr, thickness=int(fill))
        canvas = cv2.putText(canvas, tmp+drumname, org=(points[0] + 20, points[1] + 20),
                             color=(28, 255, 255), fontScale=1,
                             fontFace=cv2.FONT_HERSHEY_PLAIN)
    return canvas


def get_centroid(contour):
    m = cv2.moments(contour)
    centroid_x = m['m10'] / m['m00']
    centroid_y = m['m01'] / m['m00']
    return int(centroid_x), int(centroid_y)


def flush_centroid_record(clist, cx, cy):
    clist = np.roll(clist, -1)
    clist[-1] = np.array([[cx, cy]])
    # clist = np.append(clist, new, axis=0)
    if len(clist) == CENTROID_HISTORY_LENGTH:
        return clist
    else:
        print(f"HISTORY SIZE ISSUE{len(clist)}")
        return clist


def avg_moving_speed(clist, last_n=5):
    avg_speed = 1
    if len(clist) > last_n:
        b = clist[-last_n:]
        c = clist[-(last_n+1):-1]
        speed = np.sum((np.abs([b - c])) ** 2)
        avg_speed = speed / last_n
    else:
        return avg_speed

    return avg_speed


def avg_acce(slist, last_n=5):
    avg_acce = 0

    if len(slist) > last_n:
        b = np.array(slist[-last_n:])
        df = np.diff(b)
        avg_acce = np.sum(df) / (last_n - 1)
        return avg_acce
    else:
        return avg_acce
    return avg_acce


def moving_speed(clist):
    b = clist[1:]
    speeds = []
    for i in range(len(b)):
        s = np.sum(np.abs(b[i]-clist[i])**2)
        speeds.append(s)
    # cmean1 = np.mean(speeds[:5], axis=0)
    # cmean2 = np.mean(speeds[5:15], axis=0)
    # cmean3 = np.mean(speeds[15:], axis=0)

    # c = (cmean1 * .2 + cmean2 * .3 + cmean3 * .5) / CENTROID_HISTORY_LENGTH
    # c = (cmean1 + cmean2 + cmean3) / CENTROID_HISTORY_LENGTH
    c = np.mean(s) / CENTROID_HISTORY_LENGTH
    c = np.round(c, decimals=1)
    return c


def moving_acceleration(clist):
    b = clist[1:]
    speeds = []
    for i in range(len(b)):
        s = np.sqrt(np.sum(np.abs(b[i]-clist[i])**2))
        speeds.append(s)
    a = b[1:]
    acces = []
    for j in range(len(a)):
        t = np.sqrt(np.abs(a[j] - b[j])**2)
        acces.append(t)
    acce = np.mean(acces[:3]) * .2 + \
        np.mean(acces[3:]) * .8
    acce = acce / CENTROID_HISTORY_LENGTH
    acce = np.round(acce, decimals=2)
    return acce


def threshold_frame(frame, adaptive=False):
    if adaptive is False:
        _, thresh = cv2.threshold(frame, 0, 255, 0)
    else:
        thresh = cv2.adaptiveThreshold(frame, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 15, 1)
    return thresh


def put_hsv_text(frame, active_channel, lower_range, upper_range):
    hsv_pairs = [f"({active_channel})", f"{lower_range}", f"{upper_range}"]
    hsv_str = "\n".join(hsv_pairs)
    tcolor = (120, 120, 120)

    new_frame = cv2.putText(frame, hsv_pairs[0], org=(0, 24),
                            color=tcolor, fontScale=1,
                            fontFace=cv2.FONT_HERSHEY_PLAIN)
    new_frame = cv2.putText(new_frame, hsv_pairs[1], org=(0, 48),
                            color=tcolor,
                            fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN)
    new_frame = cv2.putText(new_frame, hsv_pairs[2], org=(0, 72),
                            color=tcolor,
                            fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN)
    text_size, baseline = cv2.getTextSize(hsv_str, fontScale=1,
                                          fontFace=0, thickness=1)
    return new_frame
