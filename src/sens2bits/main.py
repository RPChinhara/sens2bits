from __future__ import annotations

from .alloc import main as run_allocation
from .sens import main as run_sensitivity


def main() -> None:
    run_sensitivity()
    run_allocation()
