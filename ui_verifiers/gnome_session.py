import asyncio
import contextlib
import os
import signal
from Xlib import display, X
from PIL import Image
from typing import Optional, Dict
from .computer import Computer


class GnomeSession(Computer):

    def __init__(self, display: int, screen_width: int = 1280, screen_height: int = 800, color_depth: int = 24):
        self._display = display
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._color_depth = color_depth

        self._xvfb_proc: Optional[asyncio.subprocess.Process] = None
        self._gnome_proc: Optional[asyncio.subprocess.Process] = None

    def _env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env.update({
            "DISPLAY": f":{self._display}",
            "GNOME_SHELL_SESSION_MODE": "ubuntu",
            "XDG_SESSION_TYPE": "x11",
            "XDG_CURRENT_DESKTOP": "ubuntu:GNOME",
        })
        return env

    async def start(self):
        if self._xvfb_proc is not None:
            return

        screen_arg = f"{self._screen_width}x{self._screen_height}x{self._color_depth}"

        self._xvfb_proc = await asyncio.create_subprocess_exec(
            "Xvfb", f":{self._display}", "-screen", "0", screen_arg, "-nolisten", "tcp",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # Small delay to allow Xvfb to initialize
        await asyncio.sleep(0.5)

        if self._gnome_proc is None:
            self._gnome_proc = await asyncio.create_subprocess_exec(
                "dbus-run-session", "gnome-session",
                env=self._env(),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

    async def stop(self):
        # Stop GNOME session first
        if self._gnome_proc is not None:
            try:
                os.kill(self._gnome_proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self._gnome_proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    os.kill(self._gnome_proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    with contextlib.suppress(Exception):
                        await self._gnome_proc.wait()
            finally:
                self._gnome_proc = None

        # Then stop Xvfb
        if self._xvfb_proc is not None:
            try:
                os.kill(self._xvfb_proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self._xvfb_proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    os.kill(self._xvfb_proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    with contextlib.suppress(Exception):
                        await self._xvfb_proc.wait()
            finally:
                self._xvfb_proc = None

    # --- Computer implementation

    CLICK_DELAY_MS = 120

    async def screenshot(self) -> Image:
        disp = display.Display(f":{self._display}")
        root = disp.screen().root
        geom = root.get_geometry()
        width = geom.width
        height = geom.height
        raw = root.get_image(0, 0, width, height, X.ZPixmap, 0xffffffff)
        return Image.frombytes("RGB", (width, height), raw.data, "raw", "BGRX")

    async def _run_xdotool(self, *args: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            *args,
            env={"DISPLAY": f":{self._display}"},
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        returncode = await proc.wait()
        if returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(args)} (exit {returncode})")

    async def mouse_move(self, x: int, y: int):
        await self._run_xdotool("xdotool", "mousemove", str(x), str(y))

    async def left_click(self, x: int, y: int):
        await self.mouse_move(x, y)
        await self._run_xdotool("xdotool", "click", "1")

    async def right_click(self, x: int, y: int):
        await self.mouse_move(x, y)
        await self._run_xdotool("xdotool", "click", "3")

    async def double_click(self, x: int, y: int):
        await self.mouse_move(x, y)
        await self._run_xdotool(
            "xdotool",
            "click",
            "--repeat",
            "2",
            "--delay",
            str(self.CLICK_DELAY_MS),
            "1",
        )

    async def triple_click(self, x: int, y: int):
        await self.mouse_move(x, y)
        await self._run_xdotool(
            "xdotool",
            "click",
            "--repeat",
            "3",
            "--delay",
            str(self.CLICK_DELAY_MS),
            "1",
        )


if __name__ == "__main__":
    async def main():
        gnome = GnomeSession(display=10)
        print("Starting gnome..")
        await gnome.start()
        print("Gnome started")
        await asyncio.sleep(10)
        await gnome.left_click(0, 0)
        await asyncio.sleep(30)
        await gnome.stop()
        print("Gnome stopped")

    asyncio.run(main())