
import asyncio
import pyautogui
import uvicorn
import logging
from PIL import ImageGrab
from fastapi import FastAPI
from fastapi.responses import Response
from io import BytesIO

from task_manager import SimpleDataEntryTaskManager


app = FastAPI()
task_manager = SimpleDataEntryTaskManager()


@app.get("/")
async def healthcheck():
    return Response(status_code=200)


@app.get("/setup-environment")
async def setup_environment():
    try:
        await task_manager.setup()
        return Response(status_code=200)
    except Exception as e:
        return Response(status_code=500)


@app.get("/reset-environment")
async def reset_environment():
    try:
        await task_manager.reset()
        return Response(status_code=200)
    except Exception as e:
        return Response(status_code=500)


@app.get("/progress")
async def get_progress():
    num_correct_submissions, num_incorrect_submissions = task_manager.get_progress()
    return {
        "num_correct_submissions": num_correct_submissions,
        "num_incorrect_submissions": num_incorrect_submissions
    }


@app.get("/act")
async def perform_action(action_type: str, x: int, y: int, delay: float=2.0):
    if action_type == "left_click":
        pyautogui.click(x, y, button='left')
    elif action_type == "right_click":
        pyautogui.click(x, y, button='right')
    else:
        return {"error": "Invalid action_type. Use 'left_click' or 'right_click'."}

    await asyncio.sleep(delay)

    # Capture the screenshot after the click action
    screenshot = ImageGrab.grab()

    # Convert the screenshot to bytes
    img_buffer = BytesIO()
    screenshot.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    return Response(img_bytes, media_type="image/png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=60*5)
