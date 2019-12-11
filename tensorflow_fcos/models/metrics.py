class Metrics:
    def __init__(self, name=''):
        self.name = name
        self.state = 0
        self.count = 0
        self.metric_value = 0

    def update_state(self, value):
        self.state.append(value)
        self.state += value
        self.count += 1

    def result(self):
        self.metric_value = self.state / self.count
        return self.metric_value

    def reset_states(self):
        self.state = []
        self.metric_value = 0
