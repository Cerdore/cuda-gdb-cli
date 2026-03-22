"""
CUDA Exception Analyzer - CUDA_EXCEPTION mapping and SASS disassembly.

Implements cuda_analyze_exception tool handler per API schema section 2.4.
"""

import re
import gdb


class CudaExceptionAnalyzer:
    """
    CUDA Warp-level exception analyzer.
    Maps hardware exception codes to structured semantic descriptions
    and extracts the precise SASS instruction that triggered the exception.
    """

    # CUDA_EXCEPTION enum to semantic description mapping
    EXCEPTION_MAP = {
        "CUDA_EXCEPTION_1": {
            "name": "Lane Illegal Address",
            "severity": "critical",
            "description": "A single lane within the warp attempted to access "
                          "an illegal memory address.",
            "common_causes": [
                "Array index out of bounds",
                "Dereferencing a null or dangling pointer",
                "Stack buffer overflow in device code"
            ]
        },
        "CUDA_EXCEPTION_2": {
            "name": "Lane User Stack Overflow",
            "severity": "critical",
            "description": "A lane's call stack exceeded the allocated stack size.",
            "common_causes": [
                "Deep recursion in device code",
                "Large local arrays exceeding stack allocation",
                "Insufficient cudaLimitStackSize setting"
            ]
        },
        "CUDA_EXCEPTION_3": {
            "name": "Device Hardware Stack Overflow",
            "severity": "critical",
            "description": "The hardware call/return stack overflowed.",
            "common_causes": [
                "Extremely deep function call chains",
                "Recursive kernel launches (Dynamic Parallelism)"
            ]
        },
        "CUDA_EXCEPTION_4": {
            "name": "Warp Illegal Instruction",
            "severity": "critical",
            "description": "The warp encountered an illegal or undefined instruction.",
            "common_causes": [
                "Corrupted device code",
                "JIT compilation failure",
                "Architecture mismatch (running SM_80 code on SM_70 device)"
            ]
        },
        "CUDA_EXCEPTION_5": {
            "name": "Warp Out-of-Range Address",
            "severity": "critical",
            "description": "The warp attempted to access an address outside "
                          "any valid memory region.",
            "common_causes": [
                "Accessing freed device memory (use-after-free)",
                "Integer overflow in address calculation",
                "Uninitialized pointer dereference"
            ]
        },
        "CUDA_EXCEPTION_6": {
            "name": "Warp Misaligned Address",
            "severity": "warning",
            "description": "The warp performed a memory access that was not "
                          "properly aligned for the data type.",
            "common_causes": [
                "Casting pointers between types with different alignment requirements",
                "Packed struct access without proper alignment attributes"
            ]
        },
        "CUDA_EXCEPTION_7": {
            "name": "Warp Invalid Address Space",
            "severity": "critical",
            "description": "The warp attempted to access memory in an invalid "
                          "address space (e.g., shared memory address used as global).",
            "common_causes": [
                "Passing shared memory pointer to a function expecting global memory",
                "Address space confusion in generic pointer operations"
            ]
        },
        "CUDA_EXCEPTION_8": {
            "name": "Warp Invalid Program Counter",
            "severity": "critical",
            "description": "The warp's program counter jumped to an invalid address.",
            "common_causes": [
                "Corrupted function pointer",
                "Virtual function table corruption",
                "Stack smashing overwriting return address"
            ]
        },
        "CUDA_EXCEPTION_14": {
            "name": "Warp Illegal Address",
            "severity": "critical",
            "description": "Any lane within the warp accessed an illegal memory "
                          "address. This is the most common CUDA memory error.",
            "common_causes": [
                "Global memory buffer overflow",
                "Accessing device memory after cudaFree()",
                "Race condition corrupting pointer values",
                "Incorrect grid/block dimension causing out-of-bounds thread indices"
            ]
        },
        "CUDA_EXCEPTION_32": {
            "name": "Warp Shared Memory Issue",
            "severity": "critical",
            "description": "An uncorrectable error occurred during cluster-level "
                          "shared memory access.",
            "common_causes": [
                "Hardware ECC error in shared memory",
                "Cluster-level distributed shared memory access violation"
            ]
        },
        "CUDA_EXCEPTION_35": {
            "name": "Warp User Stack Overflow",
            "severity": "critical",
            "description": "The warp's user-level stack overflowed during dynamic "
                          "allocation or deep call chains.",
            "common_causes": [
                "Recursive device functions without proper depth limits",
                "alloca() or variable-length arrays exceeding stack space",
                "Insufficient per-thread stack size (cudaDeviceSetLimit)"
            ]
        },
    }

    @staticmethod
    def analyze() -> dict:
        """
        Analyze current CUDA exception state.

        Returns complete exception context including:
        - Exception type and semantic description
        - errorpc and pc values
        - Precise SASS instruction that triggered the exception
        - Current focus coordinates
        - Possible root cause hints
        """
        result = {}

        # 1. Get errorpc (precise program counter at exception trigger point)
        try:
            errorpc = gdb.parse_and_eval("$errorpc")
            result["errorpc"] = hex(int(errorpc))
        except gdb.error:
            return {
                "status": "no_exception",
                "message": "No CUDA exception detected in current context. "
                           "$errorpc is not available."
            }

        # 2. Get current pc
        try:
            pc = gdb.parse_and_eval("$pc")
            result["pc"] = hex(int(pc))
        except gdb.error:
            result["pc"] = None

        # 3. Disassemble around errorpc to locate triggering SASS instruction
        try:
            disasm_output = gdb.execute(
                f"disassemble {result['errorpc']},{result['errorpc']}+32",
                to_string=True
            )
            faulting_instruction = None
            for line in disasm_output.split('\n'):
                # Instructions at errorpc are typically marked with *> or =>
                if '*>' in line or '=>' in line:
                    faulting_instruction = line.strip()
                    break
            if faulting_instruction is None and disasm_output.strip():
                # Use first instruction as approximation
                lines = [l.strip() for l in disasm_output.split('\n') if l.strip()]
                if lines:
                    faulting_instruction = lines[0]
            result["faulting_instruction"] = faulting_instruction
        except gdb.error:
            result["faulting_instruction"] = None

        # 4. Detect exception type
        try:
            exception_output = gdb.execute("info cuda exception", to_string=True)
            exception_code = CudaExceptionAnalyzer._parse_exception_code(
                exception_output
            )
            if exception_code and exception_code in CudaExceptionAnalyzer.EXCEPTION_MAP:
                exception_info = CudaExceptionAnalyzer.EXCEPTION_MAP[exception_code]
                result["exception_code"] = exception_code
                result["exception_name"] = exception_info["name"]
                result["severity"] = exception_info["severity"]
                result["description"] = exception_info["description"]
                result["common_causes"] = exception_info["common_causes"]
            else:
                result["exception_code"] = exception_code or "UNKNOWN"
                result["description"] = "Unknown CUDA exception type"
        except gdb.error:
            result["exception_code"] = "DETECTION_FAILED"

        # 5. Attach current focus coordinates
        try:
            focus_output = gdb.execute("cuda kernel block thread", to_string=True)
            result["focus_at_exception"] = focus_output.strip()
        except gdb.error:
            pass

        # 6. Try to get key register snapshot (for address calculation reverse inference)
        try:
            key_registers = {}
            for reg_name in ["$R0", "$R1", "$R2", "$R3", "$R4", "$R5"]:
                try:
                    val = gdb.parse_and_eval(reg_name)
                    key_registers[reg_name.replace("$", "")] = hex(int(val))
                except gdb.error:
                    break
            if key_registers:
                result["key_registers_snapshot"] = key_registers
        except gdb.error:
            pass

        result["status"] = "exception_detected"
        return result

    @staticmethod
    def _parse_exception_code(output: str) -> str:
        """Parse exception code from 'info cuda exception' output."""
        match = re.search(r'CUDA_EXCEPTION_(\d+)', output)
        if match:
            return f"CUDA_EXCEPTION_{match.group(1)}"
        return None


# Tool handler registration
TOOL_HANDLER = CudaExceptionAnalyzer.analyze