import os
import contextlib
import asyncio
import pandas as pd
from ui_verifiers.session import GnomeSession
from playwright.async_api import Playwright, async_playwright, Error



class SimpleDataEntrySession(GnomeSession):

    def __init__(self, screen_width: int = 1280, screen_height: int = 800, **kwargs):
        super().__init__(
            screen_width=screen_width, 
            screen_height=screen_height,
            **kwargs
        )
        self._screen_width = screen_width
        self._screen_height = screen_height

    async def start(self):
        for attempt in range(3):
            try:
                await super().start()

                self._playwright = await async_playwright().start()

                # Open data sheet browser in left half of the primary screen
                self._data_sheet_browser = DataSheetBrowser(
                    self._playwright,
                    self._display,
                    location=(0, 0),
                    size=(round(self._screen_width / 2), self._screen_height)
                )
                await self._data_sheet_browser.start()

                # Open form browser in right half of the screen
                self._form_browser = FormBrowser(
                    self._playwright,
                    self._display,
                    location=(round(self._screen_width / 2), 0),
                    size=(round(self._screen_width / 2), self._screen_height)
                )
                await self._form_browser.start()

                # success
                return
            except Exception as exc:
                # best-effort cleanup before retry
                with contextlib.suppress(Exception):
                    await self.stop()
                if attempt < 2:
                    await asyncio.sleep(1.0)
                else:
                    raise

    async def stop(self):
        # Close Playwright browser/resources.
        with contextlib.suppress(Exception):
            if self._data_sheet_browser is not None:
                await self._data_sheet_browser.stop()
        with contextlib.suppress(Exception):
            if self._form_browser is not None:
                await self._form_browser.stop()
        with contextlib.suppress(Exception):
            if self._playwright is not None:
                await self._playwright.stop()
        
        self._playwright = None
        self._data_sheet_browser = None
        self._form_browser = None

        await super().stop()


class DataSheetBrowser:

    def __init__(self, playwright: Playwright, display: int, location: tuple[int, int], size: tuple[int, int]):
        self._playwright = playwright
        self._display = display
        self._location = location
        self._size = size
        self._browser = None
        self._context = None
        self._page = None

    async def start(self):
        self._browser = await self._playwright.chromium.launch(
            headless=False, 
            args=[
                f'--window-position={self._location[0]},{self._location[1]}',
                f"--window-size={self._size[0]},{self._size[1]}",
            ],
            env={
                'DISPLAY': f':{self._display}',
            }
        )
        self._context = await self._browser.new_context(no_viewport=True)
        self._page = await self._context.new_page()
        await self._page.goto("https://docs.google.com/spreadsheets/d/1wVwDmyx01J5_XSzdgkmxvOOHPLmZmsnq1YJ636XkwIA")

    async def stop(self):
        await self._browser.close()
        self._page = None
        self._context = None
        self._browser = None


class FormBrowser:
    def __init__(self, playwright: Playwright, display: int, location: tuple[int, int], size: tuple[int, int]):
        self._sde_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data.csv"))
        self._playwright = playwright
        self._display = display
        self._location = location
        self._size = size
        self._browser = None
        self._context = None
        self._page = None

        self.num_correct_submissions = 0
        self.num_incorrect_submissions = 0

    async def start(self):
        self._browser = await self._playwright.chromium.launch(
            headless=False, 
            args=[
                f'--window-position={self._location[0]},{self._location[1]}',
                f"--window-size={self._size[0]},{self._size[1]}"
            ],
            env={
                'DISPLAY': f':{self._display}',
            }
        )
        self._context = await self._browser.new_context(no_viewport=True)
        self._page = await self._context.new_page()
        await self._page.goto("https://docs.google.com/forms/d/e/1FAIpQLSef9VSfp3ISD7jr5Kgxq2UDibrT82vUEilN8vIrhCIfH5YfQQ/viewform")
        self._page.on("request", self._on_form_browser_request)

    async def stop(self):
        await self._browser.close()
        self._page = None
        self._context = None
        self._browser = None

    def _on_form_browser_request(self, request):
        if request.method == "POST" and request.is_navigation_request() and "entry.1444058590" in request.post_data_json:
            f_name = request.post_data_json["entry.1444058590"]
            l_name = request.post_data_json["entry.277017468"]
            email = request.post_data_json["entry.564455939"]
            matching = self._sde_data[(self._sde_data["First name"] == f_name) & (self._sde_data["Last name"] == l_name) & (self._sde_data["Email"] == email)]
            # Check submitted data against any row
            if len(matching) > 0:
                self.last_submitted_row_idx = matching.index[0]
                self.num_correct_submissions += 1
            else:
                self.num_incorrect_submissions += 1


if __name__ == "__main__":
    import asyncio
    async def main():
        session = SimpleDataEntrySession(display=10)
        print("Session starting...")
        await session.start()
        print("Session started")
        await asyncio.sleep(2)
        img = await session.screenshot()
        img.save("screenshot.png")
        await asyncio.sleep(30)
        await session.stop()
        print("Session stopped")

    asyncio.run(main())