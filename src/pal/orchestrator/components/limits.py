from pal.orchestrator.components.cgroup_manager import ResourceLimits

# Preset resource envelopes for common agent workload sizes. cpu.max
# format is "<quota> <period>" in microseconds -- e.g. "50000 100000" =
# 50% of one CPU core.

LIGHT = ResourceLimits(memory_max="256M", cpu_max="25000 100000", pids_max="20")
STANDARD = ResourceLimits(memory_max="512M", cpu_max="50000 100000", pids_max="50")
HEAVY = ResourceLimits(memory_max="2G", cpu_max="100000 100000", pids_max="100")
