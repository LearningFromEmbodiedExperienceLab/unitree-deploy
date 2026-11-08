from __future__ import annotations

import struct

from unitree_deploy.config.defaults import WIRELESS_REMOTE_BUTTON_BITS


class RemoteCommand:
    def __init__(self) -> None:
        self.lx = self.ly = self.rx = self.ry = 0.0
        self.buttons = {name: False for name in WIRELESS_REMOTE_BUTTON_BITS}
        self.pressed_edges: set[str] = set()

    def set(self, wireless_remote) -> None:
        payload = bytes(wireless_remote)
        self.lx = struct.unpack("<f", payload[4:8])[0]
        self.rx = struct.unpack("<f", payload[8:12])[0]
        self.ry = struct.unpack("<f", payload[12:16])[0]
        self.ly = struct.unpack("<f", payload[20:24])[0]

        for name, (byte_i, bit_i) in WIRELESS_REMOTE_BUTTON_BITS.items():
            pressed = bool((int(wireless_remote[byte_i]) >> bit_i) & 1)
            if pressed and not self.buttons[name]:
                self.pressed_edges.add(name)
            self.buttons[name] = pressed

    def button_pressed(self, name: str) -> bool:
        if name not in self.pressed_edges:
            return False
        self.pressed_edges.remove(name)
        return True
