from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Any, Iterator


ContextArg = AbstractContextManager[Any] | None


@contextmanager
def ctx(*contexts: ContextArg) -> Iterator[Any | tuple[Any, ...] | None]:
    """Enter each context in ``contexts`` and yield their ``__enter__`` values.

    The helper lets callers replace nested ``with`` blocks with ``with ctx(...)``
    while gracefully ignoring ``None`` entries.
    """
    with ExitStack() as stack:
        entered: list[Any] = []
        for context in contexts:
            if context is None:
                continue
            entered.append(stack.enter_context(context))

        if not entered:
            yield None
        elif len(entered) == 1:
            yield entered[0]
        else:
            yield tuple(entered)
