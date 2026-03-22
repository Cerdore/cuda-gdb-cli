"""Thread-safe command queue for RPC command dispatch."""

import queue
import threading
from dataclasses import dataclass, field
from typing import Optional, Any, Dict


@dataclass
class CommandTask:
    """
    Encapsulates a pending RPC command.

    Attributes:
        request_id: The JSON-RPC request ID
        method: The method name being invoked
        params: The parameters for the method
        result: The result of the command (set by executor)
        error: Any error that occurred (set by executor)
        completed: Threading event to signal completion
    """

    request_id: Optional[int | str]
    method: str
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    completed: threading.Event = field(default_factory=threading.Event)


class CommandQueue:
    """
    Thread-safe queue for RPC commands.

    Provides a thread-safe mechanism for passing commands from the RPC
    listener thread to the GDB main thread for execution.
    """

    def __init__(self, maxsize: int = 0):
        """
        Initialize the command queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self._queue: queue.Queue[CommandTask] = queue.Queue(maxsize=maxsize)

    def put(self, task: CommandTask, block: bool = True, timeout: Optional[float] = None) -> None:
        """
        Enqueue a command task.

        Args:
            task: The command task to enqueue
            block: Whether to block if queue is full
            timeout: Timeout in seconds (None = wait forever)
        """
        self._queue.put(task, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[CommandTask]:
        """
        Dequeue a command task.

        Args:
            block: Whether to block if queue is empty
            timeout: Timeout in seconds (None = wait forever)

        Returns:
            The command task, or None if timeout occurs
        """
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def notify_result(self, task: CommandTask) -> None:
        """
        Signal that a task has completed.

        This is called by the GDB main thread after executing a command
        to wake up the waiting RPC listener thread.

        Args:
            task: The completed command task
        """
        self._queue.task_done()

    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Get the approximate queue size."""
        return self._queue.qsize()