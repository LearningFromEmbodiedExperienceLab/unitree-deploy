import time


class LoopTimer:
    def __init__(self, dt: float):
        self.speed = 1.0
        self.dt = float(dt)
        self.next_t = time.perf_counter() + self.dt

    def sleep(self, speed: float = 1.0) -> None:
        now = time.perf_counter()
        dt = self.dt / speed
        if speed != self.speed:
            self.speed = speed
            self.next_t = now + dt
        if self.next_t > now:
            time.sleep(self.next_t - now)
            self.next_t += dt
        else:
            self.next_t = now + dt
