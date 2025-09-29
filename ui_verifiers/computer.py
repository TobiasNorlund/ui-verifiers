import enum
import asyncio
from abc import ABC, abstractmethod


class ActionType(enum.StrEnum):
    Screenshot = "screenshot"
    MouseMove = "mouse_move"
    LeftClick = "left_click"
    RightClick = "right_click"
    DoubleClick = "double_click"
    TripleClick = "triple_click"


class Computer(ABC):

    @abstractmethod
    async def screenshot(self):
        raise NotImplementedError()

    @abstractmethod
    async def mouse_move(self, x: int, y: int):
        raise NotImplementedError()

    @abstractmethod
    async def left_click(self, x: int, y: int):
        raise NotImplementedError()

    @abstractmethod
    async def right_click(self, x: int, y: int):
        raise NotImplementedError()

    @abstractmethod
    async def double_click(self, x: int, y: int):
        raise NotImplementedError()

    @abstractmethod
    async def triple_click(self, x: int, y: int):
        raise NotImplementedError()

    async def act(self, type: ActionType, **kwargs):
        match type:
            case ActionType.Screenshot:
                await self.screenshot()
            case ActionType.MouseMove:
                await self.mouse_move(x=kwargs["x"], y=kwargs["y"])
            case ActionType.LeftClick:
                await self.left_click(x=kwargs["x"], y=kwargs["y"])
            case ActionType.RightClick:
                await self.right_click(x=kwargs["x"], y=kwargs["y"])
            case ActionType.DoubleClick:
                await self.double_click(x=kwargs["x"], y=kwargs["y"])
            case ActionType.TripleClick:
                await self.triple_click(x=kwargs["x"], y=kwargs["y"])
