import numpy as np


class Metrics:
    def __init__(self, name=''):
        self.name = name
        self.state = []
        self.metric_value = 0

    def update_state(self, value):
        self.state.append(value)
        self.metric_value = np.mean(self.state)

    def result(self):
        return self.metric_value
