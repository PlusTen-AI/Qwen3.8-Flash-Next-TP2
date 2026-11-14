"""Settings and link probe. Stdlib only: runs on the host (./start.sh, Python >= 3.11) and in the container.

Settings: serve.toml (every setting with its default, in git) < serve.local.toml (this pair of machines, not in git)
< environment Q38_<SECTION>_<KEY> < command line `--set section.key=value`. Unknown sections or keys are errors.

Link probe (run on each node at every start, nothing is stored): from an address on the direct CX7 link, the route to
it gives this node's interface and address, sysfs gives the RDMA device on that interface and its RoCE v2 GID entry
for that IPv4 address. A second rail is the other half of the same CX7 port (same NIC image GUID, same physical port
name — GB10 exposes each CX7 port as two RDMA devices) when it is up with an IPv4 address. The worker probes the
head's rail addresses in order, so rail k on both nodes is the same wire.

CLI: settings.py show | get <section.key> | probe <addr>... [--rails N] [--gid N]   (probe prints JSON)"""
import ipaddress
import json
import os
import re
import subprocess
import sys
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SettingsError(Exception):
    pass


def _typed(default, raw, where):
    """Parse a string from the environment / command line by the default's type."""
    try:
        if isinstance(default, bool):
            v = raw.strip().lower()
            if v in ("1", "true", "on", "yes"):
                return True
            if v in ("0", "false", "off", "no"):
                return False
            raise ValueError
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, list):
            return [x.strip() for x in raw.split(",") if x.strip()]
        return raw
    except ValueError:
        raise SettingsError(f"{where}: '{raw}' is not a {type(default).__name__}") from None


def patch_names(root=REPO):
    """name -> default, from engine/src/model/patches.def."""
    out = {}
    with open(os.path.join(root, "engine", "src", "model", "patches.def")) as f:
        for line in f:
            m = re.match(r"\s*Q38_PATCH\(\s*\d+\s*,\s*(\w+)\s*,\s*(true|false)", line)
            if m:
                out[m.group(1)] = m.group(2) == "true"
    return out


def load(root=REPO, overrides=(), env=None):
    """Effective settings as {section: {key: value}}. overrides: ["section.key=value", ...]."""
    env = os.environ if env is None else env
    with open(os.path.join(root, "serve.toml"), "rb") as f:
        s = tomllib.load(f)
    known = patch_names(root)

    def merge(src, where):
        for sec, kv in src.items():
            if sec not in s or not isinstance(kv, dict):
                raise SettingsError(f"{where}: unknown section [{sec}] (known: {', '.join(s)})")
            for k, v in kv.items():
                if sec == "patches":
                    if k not in known:
                        raise SettingsError(f"{where}: unknown patch '{k}' (see engine/src/model/patches.def)")
                    if not isinstance(v, bool):
                        raise SettingsError(f"{where}: patches.{k} must be true or false")
                elif k not in s[sec]:
                    raise SettingsError(f"{where}: unknown key {sec}.{k} (known: {', '.join(s[sec])})")
                elif type(v) is not type(s[sec][k]):
                    raise SettingsError(f"{where}: {sec}.{k} must be a {type(s[sec][k]).__name__}, got {v!r}")
                s[sec][k] = v

    local = os.path.join(root, "serve.local.toml")
    if os.path.exists(local):
        with open(local, "rb") as f:
            try:
                merge(tomllib.load(f), "serve.local.toml")
            except tomllib.TOMLDecodeError as e:
                raise SettingsError(f"serve.local.toml: {e}") from None
    for sec, kv in s.items():
        if sec == "patches":
            continue
        for k, d in kv.items():
            e = env.get(f"Q38_{sec.upper()}_{k.upper()}")
            if e is not None:
                kv[k] = _typed(d, e, f"Q38_{sec.upper()}_{k.upper()}")
    for o in overrides:
        if "=" not in o or "." not in o.split("=", 1)[0]:
            raise SettingsError(f"--set {o}: expected section.key=value")
        key, raw = o.split("=", 1)
        sec, k = key.split(".", 1)
        if sec == "patches":
            merge({sec: {k: _typed(True, raw, f"--set {o}")}}, "--set")
        elif sec in s and k in s[sec]:
            s[sec][k] = _typed(s[sec][k], raw, f"--set {o}")
        else:
            raise SettingsError(f"--set {o}: unknown setting {key}")
    p = s["paths"]
    for k in ("weights", "work"):
        if p[k]:
            p[k] = os.path.normpath(os.path.join(root, os.path.expanduser(p[k])))
    if p["worker_repo"]:
        p["worker_repo"] = os.path.expanduser(p["worker_repo"])
    return s


def patches_env(s):
    """The [patches] section as a Q38_PATCHES value ("+a,-b"; empty = all defaults)."""
    return ",".join(("+" if v else "-") + k for k, v in s["patches"].items())


def worker_addrs(s):
    """The worker's CX7 address(es): cluster.worker_addr, else the host part of cluster.worker."""
    a = list(s["cluster"]["worker_addr"])
    if not a and s["cluster"]["worker"]:
        a = [s["cluster"]["worker"].rsplit("@", 1)[-1]]
    return a


# ---- link probe --------------------------------------------------------------------------------------------------
class ProbeError(Exception):
    pass


def _read(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def _route(addr):
    """(interface, source address) of the direct route to addr."""
    try:
        ipaddress.IPv4Address(addr)
    except ValueError:
        raise ProbeError(f"'{addr}' is not an IPv4 address (cluster.worker_addr takes CX7 IPv4 addresses)") from None
    r = subprocess.run(["ip", "-o", "route", "get", addr], capture_output=True, text=True)
    w = r.stdout.split()
    if r.returncode or "dev" not in w:
        raise ProbeError(f"no route to {addr}: {(r.stderr or r.stdout).strip()}")
    if w[0] == "local":
        raise ProbeError(f"{addr} is this node's own address")
    if "via" in w:
        raise ProbeError(f"{addr} is reached through a gateway ({w[w.index('via') + 1]}), not a direct CX7 link")
    return w[w.index("dev") + 1], w[w.index("src") + 1] if "src" in w else None


def _ipv4(dev):
    """[(address, prefix length)] of an interface."""
    r = subprocess.run(["ip", "-o", "-4", "addr", "show", "dev", dev], capture_output=True, text=True)
    return [(m.group(1), int(m.group(2))) for m in re.finditer(r"inet (\d+\.\d+\.\d+\.\d+)/(\d+)", r.stdout)]


def _hca_of(dev):
    d = f"/sys/class/net/{dev}/device/infiniband"
    try:
        h = sorted(os.listdir(d))
    except OSError:
        h = []
    return h[0] if h else None


def _netdev_of(hca):
    try:
        return sorted(os.listdir(f"/sys/class/infiniband/{hca}/device/net"))[0]
    except (OSError, IndexError):
        return None


def _gid(hca, dev, ip, want=-1):
    """Index of the RoCE v2 GID entry of hca port 1 holding ip (IPv4-mapped) on dev."""
    p = f"/sys/class/infiniband/{hca}/ports/1"
    mapped = ipaddress.IPv6Address("::ffff:" + ip).exploded
    try:
        idx = sorted(int(i) for i in os.listdir(f"{p}/gids"))
    except OSError:
        raise ProbeError(f"{hca}: no GID table in sysfs") from None
    for i in idx:
        if want >= 0 and i != want:
            continue
        g = _read(f"{p}/gids/{i}")
        t = _read(f"{p}/gid_attrs/types/{i}")
        n = _read(f"{p}/gid_attrs/ndevs/{i}")
        if g == mapped and "v2" in t and n == dev:
            return i
    if want >= 0:
        raise ProbeError(f"{hca}: GID index {want} is not the RoCE v2 entry of {ip} on {dev} (cluster.gid_index)")
    raise ProbeError(f"{hca}: no RoCE v2 GID entry for {ip} on {dev} — is RoCE v2 enabled on this interface?")


def _rail(dev, ip, why, gid=-1):
    hca = _hca_of(dev)
    if not hca:
        raise ProbeError(f"{dev} ({why}) has no RDMA device — set cluster.worker_addr to the worker's CX7 address")
    st = _read(f"/sys/class/infiniband/{hca}/ports/1/state")
    if "ACTIVE" not in st:
        raise ProbeError(f"{hca} on {dev} ({why}) is not active: {st or 'unknown state'}")
    pre = dict(_ipv4(dev)).get(ip)
    if pre is None:
        raise ProbeError(f"{dev} has no IPv4 address {ip}")
    return {"hca": hca, "dev": dev, "ip": ip, "prefix": pre, "gid": _gid(hca, dev, ip, gid),
            "mtu": int(_read(f"/sys/class/net/{dev}/mtu", "0"))}


def _siblings(hca):
    """Other RDMA devices on the same CX7 port (same NIC image GUID and physical port name), up, with IPv4."""
    base = "/sys/class/infiniband"
    guid = _read(f"{base}/{hca}/sys_image_guid")
    ppn = _read(f"/sys/class/net/{_netdev_of(hca)}/phys_port_name")
    out = []
    for h in sorted(os.listdir(base)):
        n = _netdev_of(h)
        if h == hca or not n or _read(f"{base}/{h}/sys_image_guid") != guid:
            continue
        if _read(f"/sys/class/net/{n}/phys_port_name") != ppn or "ACTIVE" not in _read(f"{base}/{h}/ports/1/state"):
            continue
        a = _ipv4(n)
        if a:
            out.append((n, a[0][0]))
    return out


def probe(targets, rails=0, gid=-1):
    """Rails of this node toward targets (addresses on the peer). One target: the routed device + its siblings
    (rails = 0: all, 1: only the routed one). Several targets: one rail per target, in that order."""
    if not targets:
        raise ProbeError("no CX7 address given (cluster.worker_addr)")
    out = []
    for t in targets:
        dev, src = _route(t)
        out.append(_rail(dev, src, f"route to {t}", gid))
    if len(targets) == 1 and rails != 1:
        for n, ip in _siblings(out[0]["hca"]):
            out.append(_rail(n, ip, f"sibling of {out[0]['hca']}", gid))
    if rails > 0:
        out = out[:rails]
    if len({r["hca"] for r in out}) < len(out):
        raise ProbeError("two rails on the same RDMA device: " + ", ".join(r["hca"] for r in out))
    return out


def same_wire(a, b):
    """Rails a (one node) and b (the other) are on one subnet."""
    na = ipaddress.IPv4Network(f"{a['ip']}/{a['prefix']}", strict=False)
    return ipaddress.IPv4Address(b["ip"]) in na


def main(argv):
    if not argv or argv[0] not in ("show", "get", "probe"):
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    try:
        if argv[0] == "probe":
            rails, gid, addrs, i = 0, -1, [], 1
            while i < len(argv):
                if argv[i] in ("--rails", "--gid"):
                    v = int(argv[i + 1])
                    rails, gid = (v, gid) if argv[i] == "--rails" else (rails, v)
                    i += 2
                else:
                    addrs.append(argv[i])
                    i += 1
            print(json.dumps(probe(addrs, rails, gid)))
            return 0
        s = load()
        if argv[0] == "show":
            print(json.dumps(s, indent=1))
        else:
            sec, k = argv[1].split(".", 1)
            v = s[sec][k]
            print(",".join(v) if isinstance(v, list) else str(v).lower() if isinstance(v, bool) else v)
        return 0
    except (SettingsError, ProbeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
