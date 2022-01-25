import signal
import sys

import cv2.cv2 as cv2

from gym_single_dish_antenna import SingleDishAntenna
from motor import Motor
from point import Point
from sun import Sun
import torch


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class Satellite(Point):
    pass


class Celestial(Point):
    pass


sgms_13a = Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3)
ref_motor = Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3)
# rpm_record = []
# ref_record = []
# while sgms_13a.rpm < 1500:
#     revs = sgms_13a.spin(1500, 1, random_torque=True)
#     rpm_record.append(sgms_13a.rpm)
# while sgms_13a.rpm > 0:
#     sgms_13a.spin(0, 1, random_torque=True)
#     rpm_record.append(sgms_13a.rpm)
# while ref_motor.rpm < 1500:
#     ref_motor.spin(1500, 1)
#     ref_record.append(ref_motor.rpm)
# while ref_motor.rpm > 0:
#     ref_motor.spin(0, 1)
#     ref_record.append(ref_motor.rpm)
# plt.plot(rpm_record)
# plt.plot(ref_record)
# plt.show()
# print('Done!')

A = SingleDishAntenna(refresh_rate=1)
sun = Sun(time='2021-08-16 02:00:00')
A.add_target(sun)
A.elements.append(Sun(time='2021-08-16 02:00:00'))
A.antenna.set_ns_motor(Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3),
                       1.0 / 30000)
A.antenna.set_ew_motor(Motor(max_rpm=1500, motor_moi=20.5, load_moi=41, motor_peak_torque=23.3),
                       1.0 / 59400)
print(f'Antenna simulator started. Location: {A.antenna.name} [{A.antenna.location}]')
print(f'Frame rate: {A.refresh_rate}')
i = 0
frames = []
out = cv2.VideoWriter('Antenna.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=100,
                      frameSize=(900, 900),
                      isColor=1)
while i < 1000:
    A.step(A.action_space.sample())
    frame = A.render(mode='human')
    if i % 10 == 0:
        print(f'Frame: {i}')
        cv2.imshow('Game', frame)
        cv2.waitKey(1)
    if frame is not None:
        out.write((frame * 255).astype('uint8'))
    i += 1

cv2.destroyAllWindows()
cv2.waitKey(0)
out.release()
