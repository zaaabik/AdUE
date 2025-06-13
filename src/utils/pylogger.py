from __future__ import annotations

import logging
from collections.abc import Mapping

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,  # pylint: disable=redefined-outer-name
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        Args:
            name: The name of the logger. Default is ``__name__``.
            rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
            extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(
        self,
        level: int,
        msg: str,
        rank: int | None = None,
        *args,
        **kwargs,  # pylint: disable=keyword-arg-before-vararg
    ) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        Args:
            level: The level to log at. Look at `logging.__init__.py`
                for more information.
            msg: The message to log.
            rank: The rank to log at.
            *args: Additional args to pass to the underlying logging
                function.
            **kwargs: Any additional keyword args to pass to the
                underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            elif rank is None:
                self.logger.log(level, msg, *args, **kwargs)
            elif current_rank == rank:
                self.logger.log(level, msg, *args, **kwargs)
