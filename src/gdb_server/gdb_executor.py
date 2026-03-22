"""Safe GDB command execution with thread isolation."""

import gdb
import threading
from typing import Any, Callable, Dict, Optional


class GdbExecutor:
    """
    Safe GDB command execution with post_event dispatch.

    All GDB API calls must go through this class to ensure they
    execute on the main GDB thread, never from the RPC listener thread.
    """

    @staticmethod
    def execute_sync(
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        params: Dict[str, Any],
        timeout: float = 25.0
    ) -> Dict[str, Any]:
        """
        Execute handler in GDB main thread, wait for result.

        This method posts the handler to the GDB main thread via
        gdb.post_event() and blocks until the result is available.

        Args:
            handler: The handler function to execute (takes params, returns result)
            params: Parameters to pass to the handler
            timeout: Maximum time to wait for result in seconds

        Returns:
            {"result": ...} on success or {"error": ...} on failure
        """
        result_container: Dict[str, Any] = {}
        completed_event = threading.Event()

        def execute_on_main_thread() -> None:
            """Execute the handler on the GDB main thread."""
            try:
                result_container["result"] = handler(params)
            except Exception as e:
                result_container["error"] = {
                    "code": -32000,
                    "message": str(e),
                    "data": {"source": "gdb_executor"}
                }
            finally:
                completed_event.set()

        # Schedule execution on main thread
        gdb.post_event(execute_on_main_thread)

        # Wait for completion
        if not completed_event.wait(timeout=timeout):
            return {
                "error": {
                    "code": -32001,
                    "message": f"Command timeout after {timeout}s",
                    "data": {"source": "gdb_executor", "timeout": timeout}
                }
            }

        # Return the result or error
        if "error" in result_container:
            return result_container["error"]
        return {"result": result_container.get("result")}

    @staticmethod
    def execute_async(
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Schedule handler in GDB main thread, return immediately.

        This method posts the handler to the GDB main thread via
        gdb.post_event() and returns immediately without waiting
        for completion. Use for non-blocking commands like continue.

        Args:
            handler: The handler function to execute (takes params, returns result)
            params: Parameters to pass to the handler

        Returns:
            {"status": "running", "blocked": True} indicating async execution started
        """

        def execute_on_main_thread() -> None:
            """Execute the handler on the GDB main thread."""
            try:
                handler(params)
            except Exception:
                # Async errors are handled via notification channel
                pass

        # Schedule execution on main thread, return immediately
        gdb.post_event(execute_on_main_thread)

        return {
            "status": "running",
            "blocked": True,
            "message": "Command dispatched to main thread"
        }