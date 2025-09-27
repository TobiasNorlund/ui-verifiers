import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        # Launch a real browser window, positioned and sized
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--window-position=200,100",
                "--window-size=1200,800"
            ]
        )
        page = await browser.new_page()
        await page.goto("https://docs.google.com/spreadsheets/d/1wVwDmyx01J5_XSzdgkmxvOOHPLmZmsnq1YJ636XkwIA")

        print("Browser ready. You can interact manually now.")

        # Option 1: just sleep while the user clicks around
        await asyncio.sleep(30)

        # Once resumed, Playwright sees the live page state
        print("Title:", await page.title())
        print("URL:", page.url)

        await browser.close()

asyncio.run(main())
