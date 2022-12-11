class FiniteStateMachine:
    def __init__(self):
        self.landed = False

    def get_state(self, t: float) -> int:
        if 0 <= t and t < 0.3:
            return 0
        elif 0.3 <= t and t < 0.6:
            return 1
        elif 0.6 <= t and not self.landed:
            return 2
        elif 0.6 <= t and self.landed:
            return 3

    def set_landing(self):
        self.landed = True
