import asyncio
from session import SimpleDataEntrySession


async def main():
    session = SimpleDataEntrySession()
    await session.start()
    # Sleep indefinitely to keep the process alive
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())