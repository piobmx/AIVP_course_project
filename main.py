__author__ = "Zhangyi Wu"

import cv2
import matplotlib.pyplot as plt
import simpleaudio as sa
from datetime import datetime
import numpy as np
# from utilx import find_largest_contour, find_largest_x_contours
from utilx import *
from motion import *


window_name = "AIR INSTRUMENT"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, 200, 200)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
WINDOW_SIZE = (600, 600)

# hbg = cv2.bgsegm.createBackgroundSubtractorCNT()

centroid_record = np.zeros((CENTROID_HISTORY_LENGTH, 2), dtype=np.uint8)
centroid_x, centroid_y = int(WINDOW_SIZE[0]/2), int(WINDOW_SIZE[1]/2)
current_moving_speed = 1
speed_log = [current_moving_speed, current_moving_speed, current_moving_speed]
acce_log = [current_moving_speed, current_moving_speed, current_moving_speed]
# min_hue, max_hue = 106, 256
# min_sat, max_sat = 110, 170
# min_sat, max_sat = 50, 180  # AT NIGHT
# min_val, max_val = 100, 238

# min_hue, max_hue = 100, 140
# min_sat, max_sat = 50, 100
# min_val, max_val = 170, 230

# min_hue, max_hue = 105, 140
# min_sat, max_sat = 48, 132
# min_val, max_val = 71, 236
min_hue, max_hue = 79, 119
min_sat, max_sat = 7, 255
min_val, max_val = 133, 228
hsv_rolling = np.array([True, False, False])
active_channel = 0
hsv_m = np.array([[min_hue, max_hue],
                  [min_sat, max_sat],
                  [min_val, max_val]])

th_ones = np.ones((600, 600))
puttext = True
speed_plot = np.zeros((600, 600, 3), np.uint8)
mhi_histories = []
clicked_pixels = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixels.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x)+", "+str(y)

        n = datetime.now()
        fn = str(n.second)+str(n.minute)

        color = canvas[x-3:x+3, y-3:y+3]
        color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
        color = np.mean(color, axis=0)
        color = str(color) + "\n" + fn + "\n"

        with open("hand_color.txt", "a+") as f:
            f.write(color)


hist_h, hist_w = 200, 600
histSize = 256
_, zero_frame = cap.read()
zero_frame = cv2.resize(zero_frame, WINDOW_SIZE)
zero_hsv = cv2.cvtColor(zero_frame, cv2.COLOR_RGB2HSV)
v_values = cv2.calcHist(zero_hsv, [2], None, [histSize],
                        (0, 256), accumulate=False)
histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
bin_w = int(round(hist_w / 256))

# print(v_values)
print(np.max(v_values))
bin_max = np.where(v_values == np.max(v_values))
print(bin_max)

write_to_vid = True
show_speed = False
show_hist = False
adaptive = True

if write_to_vid:
    out = cv2.VideoWriter('outpy.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          10, WINDOW_SIZE)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, WINDOW_SIZE)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    if show_hist:
        v_values = cv2.calcHist(hsv, [2], None, [histSize],
                                (0, 256), accumulate=False)
        histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        for i in range(1, histSize):
            histImage = cv2.line(histImage, (bin_w*(i-1),
                                             hist_h - int((v_values[i-1]))),
                                 (bin_w * (i), hist_h - int((v_values[i]))),
                                 (255, 0, 0), thickness=1)
        cv2.imshow("HISTOGRAM", histImage)

    lower_range = np.array([hsv_m[0][0], hsv_m[1][0], hsv_m[2][0]])
    upper_range = np.array([hsv_m[0][1], hsv_m[1][1], hsv_m[2][1]])
    skin_mask = cv2.inRange(hsv, lower_range, upper_range)

    frame = cv2.bitwise_and(frame, frame, mask=skin_mask)

    # gray_frame = cv2.cvtColor(skin_mask, cv2.COLOR_RGB2GRAY)
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

    # ret,thresh1 = cv2.threshold(skin_mask, 70, 255,
    #                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if adaptive is False:
        _, thresh = cv2.threshold(skin_mask, 0, 100, 0)
    else:
        thresh = cv2.adaptiveThreshold(skin_mask, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 2)

    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
    # lk = find_largest_x_contours(contours, 1)
    lk = [find_largest_contour(contours)]
    hull = []
    hull.append(cv2.convexHull(lk[0]))

    x, y, w, h = 0, 0, 0, 0
    if len(lk) > -1:
        x, y, w, h = cv2.boundingRect(lk[0])
        h = w
        centroid_x, centroid_y = get_centroid(lk[0])
        # centroid_x, centroid_y = int(x + w / 2), int(y + h / 2)
        # centroid_x, centroid_y = int(x), int(y)
        centroid_record = flush_centroid_record(centroid_record,
                                                centroid_x, centroid_y)

    img = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

    canvas = frame # : thresh /
    canvas = draw_rectangles(canvas)

    cv2.drawContours(canvas, lk, 0, (123, 233, 48), thickness=3)
    cv2.drawContours(canvas, hull, -1, (250, 50, 0), 3)
    # cv2.circle(canvas, (centroid_x, centroid_y), 12, (0, 10, 190))
    # cv2.rectangle(canvas, (x, y), (x+w, y+h), (26, 128, 12), 2)
    if x != 0:
        if_overlap(canvas, [x, y, w, h], speed=speed_log[-1])
    # cv2.drawContours(canvas, [approx], 0, (0, 255, 0), 3)

    if puttext:
        pass
        # canvas = put_hsv_text(canvas, active_channel, lower_range, upper_range)
    # canvas = cv2.addWeighted(frame, 1, canvas, 1, 0)

#     canvas = cv2.putText(canvas, f"{current_moving_speed}", org=(10, 224),
#                         color=(28, 255, 255), fontScale=1,
#                         fontFace=cv2.FONT_HERSHEY_PLAIN)
#
    cv2.imshow(window_name, canvas)
    if write_to_vid:
        pass
        # out.write(canvas)
    # cv2.setMouseCallback(window_name, click_event)

    tmp = np.arange(0, (len(acce_log)*3), 3)
    sl = np.array(speed_log) + 300
    # sl = np.array(acce_log)*4 + 300
    pts = (np.array((tmp, sl)).T).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))

    if show_speed:
        cv2.line(speed_plot, (0, 490), (600, 490), color=(20, 220, 190))
        cv2.polylines(speed_plot, [pts], isClosed=False, color=(0, 125, 12),
                      thickness=1)

        cv2.imshow("speed", speed_plot)

    active_channel = np.where(hsv_rolling)[0][0]
    current_moving_speed = avg_moving_speed(centroid_record)
    current_acce = avg_acce(speed_log)

    acce_log.append(current_acce)
    if len(acce_log) >= 200:
        acce_log = acce_log[-200:]

    speed_log.append(current_moving_speed)
    if len(speed_log) >= 200:
        speed_log = speed_log[-200:]
    speed_plot = np.zeros((600, 600, 3), np.uint8)

    key = cv2.waitKey(1)
    # print('You pressed %d (0x%x), 2LSB: %d (%s)' %
    #       (key, key, key % 2**16,
    #       repr(chr(key%256)) if key%256 < 128 else '?'))
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        n = datetime.now()
        fn = str(n.second)+str(n.minute)
        cv2.imwrite(f"{fn}screenshot.png", canvas)
        cv2.imwrite(f"{fn}histscreenshot.png", histImage)

    elif key & 0xFF == ord("k"):
        hsv_m[active_channel][0] += 1
    elif key & 0xFF == ord("j"):
        hsv_m[active_channel][0] -= 1
    elif key & 0xFF == ord("l"):
        hsv_m[active_channel][1] += 1
    elif key & 0xFF == ord("h"):
        hsv_m[active_channel][1] -= 1
    elif key & 0xFF == ord("x"):
        puttext = not puttext

    elif key & 0xFF == ord('\x03'):
        hsv_rolling = np.roll(hsv_rolling, 1)
    elif key & 0xFF == ord('\x02'):
        hsv_rolling = np.roll(hsv_rolling, -1)


# When everything done, release the capture
cap.release()
if write_to_vid:
    out.release()
cv2.destroyAllWindows()
