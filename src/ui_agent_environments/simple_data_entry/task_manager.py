import asyncio
import screeninfo
import pandas as pd
import os
from playwright.async_api import Playwright, async_playwright


class SimpleDataEntryTaskManager:

    def __init__(self,):
        self._playwright = None
        self._browser = None
        self._data_sheet_browser = None
        self._form_browser = None

    async def setup(self):
        self._playwright = await async_playwright().start()

        # Get primary screen
        primary_screen = screeninfo.get_monitors()[0]
        scaling_factor = 2

        # Open data sheet browser in left half of the primary screen
        self._data_sheet_browser = DataSheetBrowser(
            self._playwright, 
            location=(0, 0),
            size=(round(primary_screen.width / scaling_factor / 2), round(primary_screen.height / scaling_factor))
        )
        await self._data_sheet_browser.setup()

        # Open form browser in right half of the screen
        self._form_browser = FormBrowser(
            self._playwright, 
            location=(round(primary_screen.width / scaling_factor / 2), 0),
            size=(round(primary_screen.width / scaling_factor / 2), round(primary_screen.height / scaling_factor))
        )
        await self._form_browser.setup()

    async def reset(self):
        # Close Playwright browser/resources.
        if not self._browser and not self._playwright:
            return

        await self._data_sheet_browser.reset()
        await self._form_browser.reset()
        await self._playwright.stop()
        
        self._playwright = None
        self._data_sheet_browser = None
        self._form_browser = None

    @property
    def is_setup(self) -> bool:
        return self._playwright is not None

    def get_progress(self):
        return self._form_browser.num_correct_submissions, self._form_browser.num_incorrect_submissions


class DataSheetBrowser:

    def __init__(self, playwright: Playwright, location: tuple[int, int], size: tuple[int, int]):
        self._playwright = playwright
        self._location = location
        self._size = size
        self._browser = None
        self._context = None
        self._page = None

    async def setup(self):
        self._browser = await self._playwright.chromium.launch(headless=False, args=[
            f'--window-position={self._location[0]},{self._location[1]}',
        ])
        self._context = await self._browser.new_context(viewport={"width": self._size[0], "height": self._size[1]})
        self._page = await self._context.new_page()
        await self._page.goto("https://docs.google.com/spreadsheets/d/1wVwDmyx01J5_XSzdgkmxvOOHPLmZmsnq1YJ636XkwIA")

    async def reset(self):
        await self._browser.close()
        self._page = None
        self._context = None
        self._browser = None


class FormBrowser:
    def __init__(self, playwright: Playwright, location: tuple[int, int], size: tuple[int, int]):
        self._sde_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data.csv"))
        self._playwright = playwright
        self._location = location
        self._size = size
        self._browser = None
        self._context = None
        self._page = None

        self.num_correct_submissions = 0
        self.num_incorrect_submissions = 0

    async def setup(self):
        self._browser = await self._playwright.chromium.launch(headless=False, args=[
            f'--window-position={self._location[0]},{self._location[1]}',
        ])
        self._context = await self._browser.new_context(viewport={"width": self._size[0], "height": self._size[1]})
        self._page = await self._context.new_page()
        await self._page.goto("https://docs.google.com/forms/d/e/1FAIpQLSef9VSfp3ISD7jr5Kgxq2UDibrT82vUEilN8vIrhCIfH5YfQQ/viewform")
        self._page.on("request", self._on_form_browser_request)

    async def reset(self):
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
    async def main():
        sde_task_manager = SimpleDataEntryTaskManager()
        await sde_task_manager.setup()
        await asyncio.sleep(100)
        await sde_task_manager.reset()
    asyncio.run(main())