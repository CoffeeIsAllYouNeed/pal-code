"""Esc stops Pal while it works (a model reply, a check running), like Ctrl+C, and the box comes back.

While a spinner runs, a small thread watches the keyboard. A lone Esc (not the start of an arrow key's escape sequence) interrupts the main
thread, which the request code already handles as "Stopped.". Ctrl+C keeps working as before. Only in a real terminal.
"""
from __future__ import annotations

import _thread
import os
import sys
import threading
import time


class EscToStop:
    def __init__(self, enabled: bool | None = None):
        self.enabled = (sys.stdin.isatty() and sys.stdout.isatty()) if enabled is None else enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved: list | None = None
        self.pressed = False

    def __enter__(self) -> EscToStop:
        if not self.enabled:
            return self
        if os.name != "nt":
            try:
                import termios
                import tty
                fd = sys.stdin.fileno()
                self._saved = termios.tcgetattr(fd)
                tty.setcbreak(fd)                       # keys arrive one by one; Ctrl+C still sends an interrupt
            except Exception:  # noqa: BLE001  (not a real terminal after all)
                self.enabled = False
                return self
        self._thread = threading.Thread(target=self._watch, daemon=True, name="pal-esc")
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(0.5)
        if self._saved is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
            except Exception:  # noqa: BLE001
                pass

    def _fire(self) -> None:
        self.pressed = True
        self._stop.set()
        _thread.interrupt_main()

    def _watch(self) -> None:
        if os.name == "nt":
            import msvcrt
            while not self._stop.is_set():
                if msvcrt.kbhit():  # type: ignore[attr-defined]
                    ch = msvcrt.getwch()  # type: ignore[attr-defined]
                    if ch == "\x1b":
                        self._fire()
                        return
                    if ch in ("\x00", "\xe0") and msvcrt.kbhit():  # type: ignore[attr-defined]
                        msvcrt.getwch()  # type: ignore[attr-defined]   # the second half of an arrow or function key
                time.sleep(0.05)
            return
        import select
        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            ch = os.read(fd, 1)
            if ch != b"\x1b":
                continue
            more, _, _ = select.select([fd], [], [], 0.05)      # an arrow key sends ESC [ A at once; a lone Esc sends nothing after
            if more:
                os.read(fd, 8)
                continue
            self._fire()
            return
