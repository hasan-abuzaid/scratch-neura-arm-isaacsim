
from __future__ import annotations

import asyncio
import math


async def wait(seconds: float) -> None:
    """
    Wartet eine bestimmte Anzahl von Sekunden.

    Parameter:
        seconds: Wartezeit in Sekunden.

    Die Funktion ist asynchron und blockiert
    nicht den Hauptthread von Isaac Sim.
    """

    seconds = float(seconds)

    if not math.isfinite(seconds):
        raise ValueError(
            "Die Wartezeit muss eine endliche Zahl sein."
        )

    if seconds < 0:
        raise ValueError(
            "Die Wartezeit darf nicht negativ sein."
        )

    print(f"[Robot] Wait gestartet: {seconds:g} Sekunden")

    await asyncio.sleep(seconds)

    print(f"[Robot] Wait abgeschlossen: {seconds:g} Sekunden")
