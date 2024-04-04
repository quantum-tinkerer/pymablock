from time import perf_counter as time
from collections import defaultdict


class Timer:
    def __init__(self):
        self.times = defaultdict(float)

    def __call__(self, name):
        self.current_name = name
        return self

    def __enter__(self):
        self.t0 = time()
        return self

    def __exit__(self, *args):
        if self.current_name not in self.times:
            self.times[self.current_name] = []
        self.times[self.current_name].append(time() - self.t0)
        return False
