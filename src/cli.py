"""CUDA-GDB-CLI - CLI entry point for AI Agent debugging.

This module provides the command-line interface that AI Agents call
to debug CUDA GPU programs via cuda-gdb.
"""

import click
import json
import sys
from typing import Optional

from .client import GDBClient
from .session import SessionManager
from .formatters import print_json


@click.group()
@click.version_option(version="0.1.0")
def main():
    """CUDA-GDB-CLI - AI Agent interface for CUDA debugging."""
    pass


# ============ Session Management ============

@main.command("load")
@click.option("--binary", "-b", required=True, help="Binary file path")
@click.option("--core", "-c", required=True, help="Core dump file path")
@click.option("--gdb-path", default="cuda-gdb", help="Path to cuda-gdb")
def load_cmd(binary: str, core: str, gdb_path: str):
    """Load a CUDA coredump for analysis."""
    from .launcher import launch_core
    result = launch_core(binary=binary, core=core, gdb_path=gdb_path)
    print_json(result)


@main.command("attach")
@click.option("--pid", "-p", required=True, type=int, help="Process ID to attach")
@click.option("--binary", "-b", help="Binary file path (optional)")
@click.option("--gdb-path", default="cuda-gdb", help="Path to cuda-gdb")
def attach_cmd(pid: int, binary: Optional[str], gdb_path: str):
    """Attach to a running CUDA process."""
    from .launcher import launch_attach
    result = launch_attach(pid=pid, binary=binary, gdb_path=gdb_path)
    print_json(result)


@main.command("stop")
@click.option("--session", "-s", required=True, help="Session ID")
def stop_cmd(session: str):
    """Stop a debugging session."""
    client = GDBClient(session)
    result = client.call("stop")
    print_json(result)


# ============ CPU Commands (inherited from gdb-cli) ============

@main.command("bt")
@click.option("--session", "-s", required=True, help="Session ID")
@click.option("--full", "-f", is_flag=True, help="Show full frame info")
def bt_cmd(session: str, full: bool):
    """Show backtrace."""
    client = GDBClient(session)
    result = client.call("backtrace", full=full)
    print_json(result)


@main.command("threads")
@click.option("--session", "-s", required=True, help="Session ID")
def threads_cmd(session: str):
    """List threads."""
    client = GDBClient(session)
    result = client.call("threads")
    print_json(result)


@main.command("eval-cmd")
@click.option("--session", "-s", required=True, help="Session ID")
@click.argument("expression")
def eval_cmd(session: str, expression: str):
    """Evaluate an expression."""
    client = GDBClient(session)
    result = client.call("evaluate", expression=expression)
    print_json(result)


@main.command("locals-cmd")
@click.option("--session", "-s", required=True, help="Session ID")
def locals_cmd(session: str):
    """Show local variables."""
    client = GDBClient(session)
    result = client.call("locals")
    print_json(result)


@main.command("memory")
@click.option("--session", "-s", required=True, help="Session ID")
@click.option("--address", "-a", required=True, help="Memory address")
@click.option("--count", "-c", default=16, help="Number of bytes")
def memory_cmd(session: str, address: str, count: int):
    """Read memory."""
    client = GDBClient(session)
    result = client.call("memory", address=address, count=count)
    print_json(result)


@main.command("disasm")
@click.option("--session", "-s", required=True, help="Session ID")
@click.option("--count", "-c", default=10, help="Number of instructions")
def disasm_cmd(session: str, count: int):
    """Disassemble current location."""
    client = GDBClient(session)
    result = client.call("disassemble", count=count)
    print_json(result)


@main.command("exec")
@click.option("--session", "-s", required=True, help="Session ID")
@click.argument("command")
@click.option("--safety-level", type=click.Choice(["readonly", "readwrite", "full"]), default="readonly")
def exec_cmd(session: str, command: str, safety_level: str):
    """Execute a raw GDB command."""
    client = GDBClient(session)
    result = client.call("exec", command=command, safety_level=safety_level)
    print_json(result)


# ============ CUDA Commands ============

@main.command("cuda-threads")
@click.option("--session", "-s", required=True, help="Session ID")
@click.option("--kernel", "-k", type=int, help="Filter by kernel ID")
@click.option("--block", help="Filter by block (x,y,z)")
@click.option("--limit", default=50, help="Max results")
def cuda_threads_cmd(session: str, kernel: Optional[int], block: Optional[str], limit: int):
    """List CUDA GPU threads."""
    client = GDBClient(session)
    params = {"limit": limit}
    if kernel is not None:
        params["kernel"] = kernel
    if block:
        params["block"] = block
    result = client.call("cuda_threads", **params)
    print_json(result)


@main.command("cuda-kernels")
@click.option("--session", "-s", required=True, help="Session ID")
def cuda_kernels_cmd(session: str):
    """List active CUDA kernels."""
    client = GDBClient(session)
    result = client.call("cuda_kernels")
    print_json(result)


@main.command("cuda-focus")
@click.option("--session", "-s", required=True, help="Session ID")
@click.option("--kernel", "-k", type=int, help="Kernel ID")
@click.option("--block", help="Block coordinates (x,y,z)")
@click.option("--thread", help="Thread coordinates (x,y,z)")
def cuda_focus_cmd(session: str, kernel: Optional[int], block: Optional[str], thread: Optional[str]):
    """View or switch CUDA GPU focus."""
    client = GDBClient(session)
    params = {}
    if kernel is not None:
        params["kernel"] = kernel
    if block:
        params["block"] = block
    if thread:
        params["thread"] = thread
    result = client.call("cuda_focus", **params)
    print_json(result)


@main.command("cuda-devices")
@click.option("--session", "-s", required=True, help="Session ID")
def cuda_devices_cmd(session: str):
    """Show GPU device topology."""
    client = GDBClient(session)
    result = client.call("cuda_devices")
    print_json(result)


@main.command("cuda-exceptions")
@click.option("--session", "-s", required=True, help="Session ID")
def cuda_exceptions_cmd(session: str):
    """Show CUDA exceptions."""
    client = GDBClient(session)
    result = client.call("cuda_exceptions")
    print_json(result)


@main.command("cuda-memory")
@click.option("--session", "-s", required=True, help="Session ID")
@click.option("--space", required=True, type=click.Choice(["shared", "global", "local", "generic"]))
@click.option("--expr", required=True, help="Variable or address expression")
@click.option("--type", "element_type", default="int", help="Element type")
@click.option("--count", default=10, help="Element count")
def cuda_memory_cmd(session: str, space: str, expr: str, element_type: str, count: int):
    """Read CUDA GPU memory."""
    client = GDBClient(session)
    result = client.call("cuda_memory", space=space, expr=expr, element_type=element_type, count=count)
    print_json(result)


@main.command("cuda-warps")
@click.option("--session", "-s", required=True, help="Session ID")
@click.option("--sm", type=int, help="Filter by SM")
def cuda_warps_cmd(session: str, sm: Optional[int]):
    """Show warp status."""
    client = GDBClient(session)
    params = {}
    if sm is not None:
        params["sm"] = sm
    result = client.call("cuda_warps", **params)
    print_json(result)


if __name__ == "__main__":
    main()
