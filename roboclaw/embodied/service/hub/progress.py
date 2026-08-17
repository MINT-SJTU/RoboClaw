"""Custom tqdm that reports download/upload progress to Board."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tqdm import tqdm

from roboclaw.embodied.board.channels import CH_HUB

if TYPE_CHECKING:
    from roboclaw.embodied.board import Board

_EMIT_INTERVAL = 0.5  # seconds between progress emissions


class BoardProgressBar(tqdm):
    """tqdm subclass that emits progress to a Board channel.

    Use :func:`make_tqdm_class` to create a partially-bound class
    suitable for passing as ``tqdm_class`` to ``snapshot_download``.
    """

    # Set by make_tqdm_class via class attributes
    _board: Board | None = None
    _operation: str = ""
    _last_emit: float = 0.0

    def update(self, n: int = 1) -> bool | None:
        result = super().update(n)
        if self._board and self.total:
            now = time.monotonic()
            if now - self._last_emit >= _EMIT_INTERVAL:
                self._last_emit = now
                self._board.emit_sync(CH_HUB, {
                    "operation": self._operation,
                    "progress_bytes": self.n,
                    "total_bytes": self.total,
                    "progress_percent": round(self.n / self.total * 100, 1),
                })
        return result

    def close(self) -> None:
        super().close()


def make_tqdm_class(board: Board, operation: str) -> type[BoardProgressBar]:
    """Return a tqdm class pre-bound with *board* and *operation*."""
    return type(
        "BoundBoardProgressBar",
        (BoardProgressBar,),
        {"_board": board, "_operation": operation},
    )
