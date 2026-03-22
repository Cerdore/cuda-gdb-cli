"""
Register Probe - Safe CUDA hardware register dumping with limit detection.

Implements cuda_dump_warp_registers tool handler per API schema section 2.3.
"""

import gdb


class RegisterProbe:
    """
    CUDA hardware register safe probe.
    Determines actual register allocation limit for current warp before safely collecting register state.
    """

    # Predicate registers fixed at P0-P6
    PREDICATE_REGISTER_COUNT = 7
    MAX_POSSIBLE_REGISTERS = 255

    @staticmethod
    def dump_warp_registers() -> dict:
        """
        Safely dump all allocated registers for current focus warp.

        Returns:
            {
                "status": "ok",
                "warp_info": {"device": 0, "sm": 7, "warp": 3},
                "general_registers": {"R0": "0x00000042", ...},
                "predicate_registers": {"P0": "0x1", ...},
                "special_registers": {"CC": "0x0"},
                "register_count": 64,
                "max_possible": 255
            }
        """
        try:
            # Step 1: Determine actual register allocation limit for current warp
            max_reg_index = RegisterProbe._detect_register_limit()

            # Step 2: Safely collect general registers
            general_regs = {}
            for i in range(max_reg_index + 1):
                try:
                    val = gdb.parse_and_eval(f"$R{i}")
                    general_regs[f"R{i}"] = hex(int(val))
                except gdb.error:
                    # Reached actual boundary, stop traversal
                    break

            # Step 3: Collect predicate registers (P0-P6 fixed)
            predicate_regs = {}
            for i in range(RegisterProbe.PREDICATE_REGISTER_COUNT):
                try:
                    val = gdb.parse_and_eval(f"$P{i}")
                    predicate_regs[f"P{i}"] = hex(int(val))
                except gdb.error:
                    break

            # Step 4: Collect condition code register
            special_regs = {}
            try:
                cc_val = gdb.parse_and_eval("$CC")
                special_regs["CC"] = hex(int(cc_val))
            except gdb.error:
                pass

            # Step 5: Get warp hardware coordinates
            warp_info = RegisterProbe._get_warp_info()

            return {
                "status": "ok",
                "warp_info": warp_info,
                "general_registers": general_regs,
                "predicate_registers": predicate_regs,
                "special_registers": special_regs,
                "register_count": len(general_regs),
                "max_possible": RegisterProbe.MAX_POSSIBLE_REGISTERS
            }

        except gdb.error as err:
            return {
                "status": "error",
                "message": str(err),
                "hint": "Failed to probe registers. Ensure a valid GPU "
                        "thread is in focus and the kernel is active."
            }

    @staticmethod
    def _detect_register_limit() -> int:
        """
        Detect actual register allocation limit for current warp.

        Strategy:
        1. Try to parse 'info registers system' output to determine limit
        2. Fallback: Binary search to probe maximum accessible register index
        """
        # Strategy 1: Parse info registers output
        try:
            reg_info = gdb.execute("info registers system", to_string=True)
            max_index = 0
            for line in reg_info.split('\n'):
                line = line.strip()
                # Match lines like "R123  0x..."
                if line.startswith('R') and len(line) > 1:
                    parts = line[1:].split()
                    if parts and parts[0].isdigit():
                        index = int(parts[0])
                        max_index = max(max_index, index)
            if max_index > 0:
                return max_index
        except gdb.error:
            pass

        # Strategy 2: Binary search
        low, high = 0, RegisterProbe.MAX_POSSIBLE_REGISTERS
        last_valid = 0
        while low <= high:
            mid = (low + high) // 2
            try:
                gdb.parse_and_eval(f"$R{mid}")
                last_valid = mid
                low = mid + 1
            except gdb.error:
                high = mid - 1
        return last_valid

    @staticmethod
    def _get_warp_info() -> dict:
        """Get current focus warp hardware coordinates."""
        info = {}
        try:
            # Try cuda command to get hardware info
            output = gdb.execute("cuda sm warp lane", to_string=True)
            for token in output.split():
                if token.startswith("sm"):
                    parts = token.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        info["sm"] = int(parts[1])
                elif token.startswith("warp"):
                    parts = token.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        info["warp"] = int(parts[1])
        except gdb.error:
            pass

        # Also try device info
        try:
            output = gdb.execute("cuda device", to_string=True)
            match = output.lower().find("device")
            if match >= 0:
                # Try to extract device number
                import re
                dev_match = re.search(r'device\s+(\d+)', output, re.IGNORECASE)
                if dev_match:
                    info["device"] = int(dev_match.group(1))
        except gdb.error:
            pass

        return info


# Tool handler registration
TOOL_HANDLER = RegisterProbe.dump_warp_registers