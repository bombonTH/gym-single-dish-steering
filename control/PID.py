import numpy as np
import matplotlib.pyplot as plt


class PIDController:
    def __init__(self, kp=2.0, ki=0.5, kd=0.25, tau=0.002, lim_min=-1.0, lim_max=1.0, lim_int_min=-0.5,
                 lim_int_max=0.5, period=1):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.tau = tau

        self.lim_int_min = lim_int_min
        self.lim_int_max = lim_int_max

        self.lim_min = lim_min
        self.lim_max = lim_max

        self.period = period

        self.proportional = 0.0
        self.integrator = 0.0
        self.differentiator = 0.0

        self.error = 0
        self.prev_error = 0.0
        self.prev_measurement = 0.0

        self.out = 0.0

    def update(self, setpoint: float = 0.0, measurement: float = 0.0):
        self.error = setpoint - measurement

        self.proportional = self.kp * self.error

        self.integrator = self.integrator + 0.5 * self.ki * self.period * (self.error + self.prev_error)
        self.integrator = max(min(self.integrator, self.lim_max), self.lim_min)

        self.differentiator = -(2.0 * self.kd * (measurement - self.prev_measurement)
                                + (2.0 * self.tau - self.period) * self.differentiator) / (2.0 * self.tau + self.period)

        self.out = self.proportional + self.integrator + self.differentiator
        self.out = max(min(self.out, self.lim_max), self.lim_min)

        self.prev_error = self.error
        self.prev_measurement = measurement

        return self.out


def test():
    measurements = []
    setpoints = []
    setpoint = np.random.randint(-10, 10) / 10.0
    measurement = np.random.uniform(-0.9, 0.9)
    measurement = -1.0
    period = 1
    pid = PIDController(period=period, kp=0.2, ki=0.25, kd=0.025)
    for time in range(1000):
        action = pid.update(setpoint, measurement)
        measurement = period * action + measurement / (1.0 / 0.02 * period)
        print(f'{measurement:.2f}/{setpoint:.2f}')
        if time % 100 == 0:
            setpoint = np.random.randint(-10, 10) / 10.0
        measurements.append(measurement)
        setpoints.append(setpoint)

    plt.plot(range(len(measurements)), measurements, color='b')
    plt.plot(range(len(measurements)), setpoints, color='r')
    #plt.axhline(y=setpoint, color='r', linestyle='-')
    plt.ylim(-1, 1)
    plt.show()


if __name__ == '__main__':
    test()
