import fastapi
import pydantic
import asyncio
from typing import Dict
from contextlib import asynccontextmanager
from datetime import datetime
from enum import StrEnum
from io import BytesIO
from importlib.metadata import version
from fastapi import HTTPException, Depends
from fastapi.responses import Response
from .computer import ActionType, Computer
from .session import Session
from .tasks import SESSION_TYPES


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    app.state.display_counter = 10
    app.state.session_counter = 1
    app.state.sessions = {}
    app.state.session_info = {}
    yield
    results = await asyncio.gather(
        *(session.stop() for session in app.state.sessions.values()),
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, Exception):
            print(f"Session stop error: {r}")
    

app = fastapi.FastAPI(lifespan=lifespan)


SessionType = StrEnum("SessionType", {k: k for k in SESSION_TYPES.keys()})


class SessionInfo(pydantic.BaseModel):
    id: int
    start_time: datetime
    display: int
    action_count: int = 0
    stop_time: datetime | None = None


@app.get("/")
async def root():
    return Response(content=f"UI Verifiers Server v{version("ui-verifiers")}", media_type="text/plain")


@app.get("/status")
async def status() -> Dict[int, SessionInfo]:
    return app.state.session_info


# --- Session endpoints

# Dependency: validate and return an existing session
def get_session(session_id: int) -> Session:
    session: Session | None = app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def get_session_info(session_id: int) -> SessionInfo:
    session_info: Session | None = app.state.session_info.get(session_id)
    if session_info is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_info


@app.post("/session")
async def session_create(type: SessionType = SessionType("simple_data_entry"), n: int = 1):
    created_ids: list[int] = []
    tasks = []
    for session_id in range(app.state.session_counter, app.state.session_counter + n):
        session = SESSION_TYPES[type.value](display=app.state.display_counter)
        if not hasattr(app.state, "sessions"):
            app.state.sessions = {}
        app.state.sessions[session_id] = session
        app.state.session_info[session_id] = SessionInfo(
            id=session_id,
            start_time=datetime.now(),
            display=app.state.display_counter,
        )
        created_ids.append(session_id)
        tasks.append(session.start())
        app.state.display_counter += 1
    app.state.session_counter += n

    if tasks:
        await asyncio.gather(*tasks)

    return {"session_ids": created_ids}


@app.get("/session/{session_id}")
async def session_get(session_id: int, session_info: SessionInfo = Depends(get_session_info)):
    return session_info


@app.get("/session/{session_id}/act")
async def session_act(
    session_id: int,
    action_type: ActionType,
    x: int | None = None,
    y: int | None = None,
    delay: float = 1.0,
    session: Session = Depends(get_session),
):

    await session.act(
        type=action_type,
        x=x,
        y=y
    )
    await asyncio.sleep(delay)

    info: SessionInfo = app.state.session_info[session_id]
    info.action_count += 1

    img = await session.screenshot()
    buf = BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/session/{session_id}/screenshot")
async def session_screenshot(session_id: int, session: Session = Depends(get_session)):
    img = await session.screenshot()
    buf = BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/session/{session_id}/progress")
async def session_progress(session_id: int, session: Session = Depends(get_session)):
    return session.get_progress()


@app.delete("/session/{session_id}")
async def session_delete(session_id: int, session: Computer = Depends(get_session)):
    app.state.sessions.pop(session_id, None)
    try:
        await session.stop()
    finally:
        info: SessionInfo | None = app.state.session_info.get(session_id)
        if info is not None:
            info.stop_time = datetime.utcnow()
    return {"deleted": session_id}


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

