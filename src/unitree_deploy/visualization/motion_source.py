"""Any4HDMI reference preview, independent of the publishing rate."""

import copy
import json
import threading
import time

import mujoco

from .policy_debug_viz import POLICY_DEBUG_VIZ_TOPIC


class MotionSourceVisualizer:
    def __init__(self, model, qpos):
        import viser

        from any4hdmi.scripts.viewer.contacts import ContactHighlights
        from any4hdmi.scripts.viewer.scene import create_motion_scene
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

        self.model = copy.copy(model)
        self.data = mujoco.MjData(self.model)
        self.qpos = qpos.copy()
        self.controller_qpos = None
        self.controller_received_at = 0.0
        self.controller_subscriber = None
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.updated = threading.Event()
        self.server = viser.ViserServer(label="G1 motion reference")
        try:
            self.scene = create_motion_scene(self.server, self.model)
            self.scene.camera_tracking_enabled = True
            self.scene.site_groups_visible = [False] * 6
            self.scene._sync_visibilities()
            self.scene.create_scene_gui(camera_distance=1.5 * model.stat.extent)
            if self.scene._camera_dropdown is not None:
                self.scene._camera_dropdown.remove()
                self.scene._camera_dropdown = None
            self.contacts = ContactHighlights(self.server, self.model, self.data, self.scene, live=True)
            self.scene.set_refresh_handler(self.updated.set)
            for control in (self.contacts.ground, self.contacts.self_collision):
                control.on_update(lambda _: self.updated.set())
            self.data.qpos[:] = self.qpos
            self._update_scene()
            self.controller_subscriber = ChannelSubscriber(POLICY_DEBUG_VIZ_TOPIC, String_)
            self.controller_subscriber.Init(lambda msg: self.receive_reference(msg.data), 1)
        except BaseException:
            if self.controller_subscriber is not None:
                self.controller_subscriber.Close()
            self.server.stop()
            raise
        self.thread = threading.Thread(target=self._render, name="motion-preview", daemon=True)
        self.thread.start()

    def update(self, qpos):
        with self.lock:
            self.qpos[:] = qpos
            self.updated.set()

    def receive_reference(self, payload):
        frame = json.loads(payload)
        with self.lock:
            self.controller_qpos = None
            if frame["active"]:
                pose = self.qpos.copy()
                pose[:3] = frame["root_pos"]
                pose[3:7] = frame["root_quat"]
                indices = [self.model.jnt_qposadr[self.model.joint(name).id] for name in frame["joint_names"]]
                pose[indices] = frame["joint_pos"]
                self.controller_qpos = pose
            self.controller_received_at = time.monotonic()
            self.updated.set()

    def _render(self):
        while True:
            self.updated.wait(.5)
            if self.stopped.is_set():
                return
            with self.lock:
                self.updated.clear()
                # Follow the same consumed/aligned reference as the sim-bridge ghost.
                fresh = time.monotonic() - self.controller_received_at < .5
                if fresh and self.controller_qpos is not None:
                    self.data.qpos[:] = self.controller_qpos
                else:
                    self.data.qpos[:] = self.qpos
            self._update_scene()

    def _update_scene(self):
        # Evaluate contacts at the reference pose without advancing physics.
        with self.scene._update_lock, self.server.atomic():
            mujoco.mj_forward(self.model, self.data)
            self.scene.update_from_mjdata(self.data)
            self.contacts.update(force=True)

    def close(self):
        self.controller_subscriber.Close()
        self.stopped.set()
        self.updated.set()
        self.thread.join()
        self.scene.set_refresh_handler(None)
        self.server.stop()
