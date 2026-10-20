import re
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def pinned(path):
    out = {}
    for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)(?:\s*;\s*([^\\\n]+?))?\s*\\?$", open(path).read(), re.M):
        out[canonicalize_name(m.group(1))] = (m.group(2), (m.group(3) or "").strip())
    return out

def requires(item):
    name, (ver, _) = item
    r = requests.get(f"https://pypi.org/pypi/{name}/{ver}/json", timeout=20)
    r.raise_for_status()
    return name, r.json()["info"].get("requires_dist") or []

def check(lock, env_name, env):
    pins = pinned(lock)
    with ThreadPoolExecutor(12) as ex:
        reqs = dict(ex.map(requires, pins.items()))
    missing = {}
    for name, rds in reqs.items():
        for rd in rds:
            r = Requirement(rd)
            if r.marker and not r.marker.evaluate({**env, "extra": ""}):
                continue
            dep = canonicalize_name(r.name)
            if dep not in pins:
                missing.setdefault(dep, set()).add(name)
            else:                                   # pinned, but is the pin itself limited to a platform that excludes this one?
                marker = pins[dep][1]
                if marker and not Marker(marker).evaluate(env):
                    missing.setdefault(dep + " (pinned, but only for another platform)", set()).add(name)
    print(f"{env_name}: {len(pins)} pinned packages; required but not pinned: " + (", ".join(f"{k} (needed by {', '.join(sorted(v))})" for k, v in sorted(missing.items())) or "none"))
    return missing

lock = sys.argv[1]
base = {"python_version": "3.12", "python_full_version": "3.12.0", "implementation_name": "cpython", "platform_python_implementation": "CPython"}
win = {**base, "sys_platform": "win32", "platform_system": "Windows", "os_name": "nt", "platform_machine": "AMD64"}
lin = {**base, "sys_platform": "linux", "platform_system": "Linux", "os_name": "posix", "platform_machine": "x86_64"}
mac = {**base, "sys_platform": "darwin", "platform_system": "Darwin", "os_name": "posix", "platform_machine": "arm64"}
for n, e in (("Windows", win), ("Linux", lin), ("macOS", mac)):
    check(lock, n, e)
