from . import BaseController

class Controller(BaseController):
    def __init__(self, kp, ki, kd):
        super().__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def update(self, target_lataccel, current_lataccel, state):
        error = target_lataccel - current_lataccel
        self.integral += error
        derivative = error - self.previous_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

