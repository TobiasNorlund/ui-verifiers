import fastapi

app = fastapi.FastAPI()



@app.get("/")
async def root():
    return "UI Verifiers Server"


@app.get("/status")
async def status():
    pass


# --- Session endpoints

@app.post("/session")
async def session_create(type: str = "simple_data_entry", n: int = 1):
    pass


@app.get("/session/{session_id}/act")
async def session_act(action_type: str, x: int, y: int, delay: float=2.0):
    pass


@app.get("/session/{session_id}/screenshot")
async def session_screenshot():
    pass


@app.get("/session/{session_id}/progress")
async def session_progress():
    pass


@app.delete("/session/{session_id}")
async def session_delete():
    pass


# -- VNC endpoints

@app.put("/vnc")
async def vnc_set(session: int):
    pass


@app.get("/vnc")
async def vnc_get():
    pass


@app.delete("/vnc")
async def vnc_delete():
    pass

