"""
GDB Command Executor - Thread-safe command execution with gdb.post_event().

Provides the bridge between the RPC listener thread and GDB main thread.
All GDB API calls must go through this class to maintain thread safety.
"""

import queue
import threading
import typing
from typing import Dict, Any, Callable, Optional

import gdb


class GDBExecutor:
    """
    Safe GDB command execution with post_event dispatch.

    This class provides the thread-safe bridge between the RPC listener
    thread (which handles socket I/O) and the GDB main thread (which
    can safely call gdb.execute(), gdb.parse_and_eval(), etc.).

    Thread Model:
        - RPC Listener Thread: Handles socket I/O, creates CommandTasks
        - GDB Main Thread: Executes handlers via gdb.post_event()

    The executor ensures all GDB API calls happen in the main thread
    where they are safe to execute.
    """

    def __init__(
        self,
        command_queue: queue.Queue,
        modality_guard: Any,
        timeout: float = 25.0
    ):
        """
        Initialize the GDB executor.

        Args:
            command_queue: Thread-safe queue for command tasks
            modality_guard: Modality guard instance for permission checking
            timeout: Default timeout for command execution (seconds)
        """
        self.command_queue = command_queue
        self.modality_guard = modality_guard
        self.timeout = timeout

        # State tracking
        self._is_running = False  # Target execution state

    @staticmethod
    def execute_sync(
        handler: Callable,
        params: Dict[str, Any],
        timeout: float = 25.0
    ) -> Dict[str, Any]:
        """
        Execute handler in GDB main thread, wait for result.

        This method schedules the handler in the GDB main thread
        and blocks until the result is available.

        Args:
            handler: Function to execute (receives params dict)
            params: Parameters to pass to handler
            timeout: Maximum time to wait for result

        Returns:
            {"result": ...} on success or {"error": ...} on failure
        """
        # Create a completion event
        completed = threading.Event()
        result_container: Dict[str, Any] = {}

        def execute_and_store():
            """Execute handler and store result."""
            try:
                result_container["result"] = handler(params)
            except gdb.error as e:
                result_container["error"] = {
                    "code": -32000,
                    "message": str(e),
                    "data": {"source": "gdb_internal"}
                }
            except Exception as e:
                result_container["error"] = {
                    "code": -32603,
                    "message": str(e),
                    "data": {"source": "rpc_engine"}
                }
            finally:
                completed.set()

        # Schedule execution in main thread
        gdb.post_event(execute_and_store)

        # Wait for completion
        if not completed.wait(timeout=timeout):
            return {
                "error": {
                    "code": -32001,
                    "message": f"Command timed out after {timeout}s",
                    "data": {"source": "rpc_engine"}
                }
            }

        # Return result or error
        if "error" in result_container:
            return result_container["error"]
        return result_container.get("result", {})

    @staticmethod
    def execute_async(
        handler: Callable,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Schedule handler in GDB main thread, return immediately.

        For blocking commands like 'continue', this returns immediately
        while the actual execution continues in the background.
        When the target stops, gdb.events.stop will fire.

        Args:
            handler: Function to execute (receives params dict)
            params: Parameters to pass to handler

        Returns:
            {"status": "running", "blocked": True} indicating execution started
        """
        def execute_and_notify():
            """Execute handler, update running state on completion."""
            try:
                handler(params)
            except gdb.error as e:
                # Send error notification
                gdb.events.stop.disconnect()  # Prevent duplicate notifications
            except Exception:
                pass

        # Schedule and return immediately
        gdb.post_event(execute_and_notify)

        return {
            "status": "running",
            "blocked": True,
            "message": "Target is now running. You will receive a notification when it stops."
        }

    @staticmethod
    def execute_gdb_command(command: str, to_string: bool = False) -> str:
        """
        Execute a GDB command string.

        This is the primary method for executing GDB CLI commands.
        Must be called in GDB main thread (via post_event).

        Args:
            command: GDB command string (e.g., "cuda thread 0,0,0")
            to_string: If True, capture output as string

        Returns:
            Command output if to_string=True, otherwise empty string

        Raises:
            gdb.error: If command fails
        """
        return gdb.execute(command, to_string=to_string)

    @staticmethod
    def evaluate_expression(expression: str) -> gdb.Value:
        """
        Evaluate a GDB expression.

        Used for reading variables, registers, etc.
        Must be called in GDB main thread.

        Args:
            expression: Expression to evaluate (e.g., "threadIdx.x", "$R0")

        Returns:
            gdb.Value object

        Raises:
            gdb.error: If expression cannot be evaluated
        """
        return gdb.parse_and_eval(expression)

    def set_running_state(self, is_running: bool) -> None:
        """
        Update the target running state.

        Used by execution control to track whether the target
        is currently executing (blocking) or stopped.

        Args:
            is_running: True if target is executing, False if stopped
        """
        self._is_running = is_running

    def is_running(self) -> bool:
        """
        Check if target is currently running.

        Returns:
            True if target is executing, False if stopped
        """
        return self._is_running

    @staticmethod
    def interrupt_target() -> None:
        """
        Interrupt a running target.

        Sends SIGINT to the target to stop execution.
        Must be called in GDB main thread.
        """
        gdb.execute("interrupt")

    @staticmethod
    def continue_execution() -> None:
        """
        Continue target execution.

        This is a blocking call - it will not return until
        the target hits a breakpoint, exception, or signal.
        Must be called in GDB main thread.
        """
        gdb.execute("continue")

    @staticmethod
    def step_into() -> None:
        """
        Step into function (source level).

        Must be called in GDB main thread.
        """
        gdb.execute("step")

    @staticmethod
    def step_over() -> None:
        """
        Step over function (source level).

        Must be called in GDB main thread.
        """
        gdb.execute("next")

    @staticmethod
    def step_instruction() -> None:
        """
        Step one instruction (assembly level).

        Must be called in GDB main thread.
        """
        gdb.execute("stepi")

    @staticmethod
    def finish_function() -> None:
        """
        Run until current function returns.

        Must be called in GDB main thread.
        """
        gdb.execute("finish")

    @staticmethod
    def get_selected_frame() -> gdb.Frame:
        """
        Get the currently selected frame.

        Returns:
            gdb.Frame object

        Raises:
            RuntimeError: If no frame is selected
        """
        frame = gdb.selected_frame()
        if frame is None:
            raise RuntimeError("No frame selected")
        return frame

    @staticmethod
    def get_stop_event_reason(event: Any) -> Dict[str, Any]:
        """
        Determine the reason for a stop event.

        Args:
            event: gdb.StopEvent subclass

        Returns:
            Dict with reason information
        """
        from typing import Any as AnyType

        if hasattr(event, 'breakpoints') and event.breakpoints:
            return {
                "reason": "breakpoint",
                "breakpoint_id": event.breakpoints[0].number
            }
        elif hasattr(event, 'stop_signal'):
            return {
                "reason": "signal",
                "signal_name": event.stop_signal
            }
        else:
            return {"reason": "unknown"}

    @staticmethod
    def capture_stop_event_context() -> Dict[str, Any]:
        """
        Capture full context at stop event for notifications.

        Returns:
            Dict with focus, PC, source location, etc.
        """
        context: Dict[str, Any] = {}

        # Try to get focus coordinates
        try:
            output = gdb.execute("cuda kernel block thread", to_string=True)
            context["focus_output"] = output.strip()
        except gdb.error:
            pass

        # Try to get PC
        try:
            pc = gdb.parse_and_eval("$pc")
            context["pc"] = hex(int(pc))
        except gdb.error:
            pass

        # Try to get source location
        try:
            frame = gdb.selected_frame()
            sal = frame.find_sal()
            if sal.symtab:
                context["source_location"] = {
                    "file": sal.symtab.filename,
                    "line": sal.line
                }
        except gdb.error:
            pass

        # Try to detect CUDA exception
        try:
            errorpc = gdb.parse_and_eval("$errorpc")
            context["cuda_exception"] = {
                "errorpc": hex(int(errorpc)),
                "hint": "CUDA hardware exception detected"
            }
        except gdb.error:
            pass

        return context


class CommandTask:
    """
    Encapsulates a pending RPC command task.

    This class is used to pass command information from the
    RPC listener thread to the GDB main thread via the queue.

    Attributes:
        request_id: JSON-RPC request ID
        method: Method name being executed
        params: Method parameters
        result: Result data (set by executor)
        error: Error dict (set by executor)
        completed: Event signaling task completion
    """

    def __init__(
        self,
        request_id: Optional[Any],
        method: str,
        params: Dict[str, Any]
    ):
        self.request_id = request_id
        self.method = method
        self.params = params
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[Dict[str, Any]] = None
        self.completed = threading.Event()

    def __repr__(self) -> str:
        return f"CommandTask(method={self.method}, params={self.params})"