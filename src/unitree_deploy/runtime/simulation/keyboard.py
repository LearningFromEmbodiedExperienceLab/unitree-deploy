"""Keyboard input for simulator sticks and one-shot controls."""

import threading

from pynput import keyboard

from unitree_deploy.config.defaults import SIM_REMOTE_BUTTON_KEYS


STICK_KEYS = frozenset("wasdqe")


class KeyboardState:
    """Track held controls; pass repeat information to one-shot simulator actions."""

    def __init__(self) -> None:
        self.pressed: set[str] = set()
        self.held_keys: set[str] = set()
        self.lock = threading.Lock()
        self.enabled = True
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.control_handler = None

    @staticmethod
    def _key_name(key) -> str:
        try:
            if hasattr(key, "char") and key.char:
                return key.char.lower()
        except AttributeError:
            pass
        return str(key).replace("Key.", "").lower()

    def _on_press(self, key) -> None:
        k = self._key_name(key)
        with self.lock:
            if not self.enabled:
                return
            repeat = k in self.held_keys
            self.held_keys.add(k)
            if k in STICK_KEYS or k in SIM_REMOTE_BUTTON_KEYS:
                self.pressed.add(k)

        if self.control_handler:
            self.control_handler(k, repeat=repeat)

    def _on_release(self, key) -> None:
        k = self._key_name(key)
        with self.lock:
            self.pressed.discard(k)
            self.held_keys.discard(k)

    def start(self) -> None:
        self.listener.start()

    def stop(self) -> None:
        self.listener.stop()

    def stick_keys(self) -> set[str]:
        with self.lock:
            return self.pressed & STICK_KEYS

    def active_keys(self) -> set[str]:
        with self.lock:
            return self.pressed.copy()

    def set_control_handler(self, handler) -> None:
        self.control_handler = handler

    def set_enabled(self, enabled: bool) -> None:
        with self.lock:
            self.enabled = enabled
            self.pressed.clear()
            self.held_keys.clear()
