"""Passive G1 reference preview, independent of the publishing rate."""

import threading

import mujoco


class MotionSourceVisualizer:
    def __init__(self, model, qpos):
        import viser

        from .mujoco_scene import RobotScene

        self.model = model
        self.data = mujoco.MjData(model)
        self.qpos = qpos.copy()
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.updated = threading.Event()
        self.server = viser.ViserServer(label="G1 motion reference")
        try:
            self.scene = RobotScene(
                self.server, model, show_sites=False, show_camera_frustums=False,
                visual_rgba=(0.15, 0.65, 1.0, 0.45),
            )
            self.data.qpos[:] = self.qpos
            mujoco.mj_kinematics(model, self.data)
            self.scene.update_from_mjdata(self.data)
        except BaseException:
            self.server.stop()
            raise
        self.thread = threading.Thread(target=self._render, name="motion-preview", daemon=True)
        self.thread.start()

    def update(self, qpos):
        with self.lock:
            self.qpos[:] = qpos
            self.updated.set()

    def _render(self):
        while True:
            self.updated.wait()
            if self.stopped.is_set():
                return
            with self.lock:
                self.updated.clear()
                self.data.qpos[:] = self.qpos
            mujoco.mj_kinematics(self.model, self.data)
            self.scene.update_from_mjdata(self.data)

    def close(self):
        self.stopped.set()
        self.updated.set()
        self.thread.join()
        self.scene.close()
        self.server.stop()
