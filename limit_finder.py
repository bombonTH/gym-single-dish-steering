import coordinate
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from cv2 import cv2 as cv2

# canvas = np.load('canvas.npy')
# cv2.imshow('game', canvas)
# cv2.waitKey(10000)

azimuths = []

for az in range(0, 9000):
    for el in range(0, 9000):
        xy = coordinate.ElAz(el=el / 100, az=az / 100).to_xy()
        if math.degrees(xy.y) > 80 or math.degrees(xy.x) > 50:
            continue
        else:
            azimuths.append(el / 100)
            print(f'Az: {az} - El: {el / 100} - {math.degrees(xy.y):.4f} - {math.degrees(xy.x):.4f}')
            break
print(azimuths)
azimuths = azimuths + azimuths[::-1]
azimuths = azimuths + azimuths[::-1]
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

angle = np.arange(0, 360, 0.01)
ax.plot(angle * math.pi / 180, azimuths)
ax.set_rlim(bottom=90, top=0)
ax.set_theta_offset(math.pi / 2)
plt.show()

image = (np.zeros((840, 840, 3)) * 1).astype('uint8')
h = image.shape[0]
w = image.shape[1]
center_h = int((h / 2) - 1)
center_w = int((w / 2) - 1)
radius = min(h, w) * 0.95 * 0.5
color = (50, 50, 50)

pts = np.zeros([36000, 2], dtype=np.int32)

image = cv2.circle(image, (center_h, center_w), int(radius), color=color, thickness=2)  # horizon
image = cv2.circle(image, (center_h, center_w), int(radius * 2 / 3), color=color, thickness=2)  # horizon
image = cv2.circle(image, (center_h, center_w), int(radius * 1 / 3), color=color, thickness=2)  # horizon

for angle in np.arange(0, 180, 30):
    srt_point = (
    int(center_h - radius * math.sin(math.radians(angle))), int(center_h - radius * math.cos(math.radians(angle))))
    end_point = (
    int(center_h + radius * math.sin(math.radians(angle))), int(center_h + radius * math.cos(math.radians(angle))))
    image = cv2.line(image, srt_point, end_point, color=color, thickness=2)

for step in np.arange(0, 36000, 1):
    az = step / 100
    el = (90 - azimuths[step]) / 90 * radius
    point = (int(center_h - el * math.sin(math.radians(az))), int(center_w - el * math.cos(math.radians(az))))
    pts[step] = point
pts = pts.reshape((-1, 1, 2))
image = cv2.polylines(image, pts, True, color=(0, 0, 200), thickness=2)
np.save('canvas', image)

cv2.imshow('game', image)
cv2.waitKey(10000)
