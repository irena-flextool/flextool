"""polars' jemalloc decay settings, whichever of polars and flextool was imported first.

polars links jemalloc statically.  jemalloc reads ``_RJEM_MALLOC_CONF`` once, while ``import polars`` loads the
runtime library.  polars 1.x's own values, which ``import polars`` writes into the variable, are
``dirty_decay_ms:500,muzzy_decay_ms:-1``: freed pages pass to the "muzzy" (MADV_FREE) state and stay resident, so a
large model's peak memory roughly doubles.  polars 2.x writes ``muzzy_decay_ms:1000``, which returns them after ~2 s.
FlexTool wants ``dirty_decay_ms:1000,muzzy_decay_ms:0``: a freed page is returned to the OS about a second after it
goes idle.

* polars not loaded yet: the environment variable is set (``setdefault``, so a value from the shell wins) and
  jemalloc reads it when polars loads.  ``import polars`` puts its own values in front, and the later ones win.
* polars already loaded (a Spine Toolbox console, a notebook, a script that imported polars first): FlexTool's
  values are appended to the variable, for child processes, and the same two decay values are written at runtime
  through jemalloc's ``mallctl``: the default for new arenas (``arenas.*``) and every initialised arena
  (``arena.<i>.*``).  The ``mallctl`` symbol is local to the polars library (not exported), so its address is
  computed from the ELF symbol table: load base = address of the library's exported ``PyInit_*`` function minus its
  symbol value.  The first bytes of the function at the computed address are compared with the same bytes of the
  file before the function is called; any mismatch, a non-ELF library (macOS) or a failing ``mallctl`` call prints
  one warning on stderr with the remedy and changes nothing.  A ``_RJEM_MALLOC_CONF`` set before polars was
  imported has already been read by jemalloc and wins as a whole, as it would with flextool imported first.  One
  set after polars was imported (it then lacks polars' own values in front) has its decay keys written at runtime.

Windows polars uses mimalloc, and a library without jemalloc has nothing to configure: both are a silent no-op.
"""
from __future__ import annotations

import os
import re
import sys

ENV_VAR = "_RJEM_MALLOC_CONF"
DEFAULT_CONF = "dirty_decay_ms:1000,muzzy_decay_ms:0"
DECAY_KEYS = ("dirty_decay_ms", "muzzy_decay_ms")
MALLCTL_SYMBOL = "_rjem_mallctl"  # local (not exported) in the polars library
REMEDY = (
    "FlexTool: polars was imported before flextool and its memory allocator settings could not be adjusted "
    "({reason}); without them peak memory can double. Import flextool before polars, or start Python with "
    f"{ENV_VAR}={DEFAULT_CONF}"
)
_EFAULT = 14          # mallctl on an arena index that is not initialised
_COMPARE_BYTES = 32   # bytes of the function compared between memory and file
_SHT_SYMTAB = 2
_STT_FUNC = 2

# ``import polars`` writes its own decay values in front of any value that was already set, before the runtime
# library loads; what follows them is the value set before polars was imported.
_POLARS_PREFIX = re.compile(r"dirty_decay_ms:-?\d+,muzzy_decay_ms:-?\d+(,thp:always,metadata_thp:always)?(?=,|$)")

_state = {"warned": False}


class AllocatorUnreachable(Exception):
    """jemalloc's mallctl in the loaded polars library cannot be located or called safely."""


def configure_polars_allocator() -> None:
    """Set the decay config for polars' jemalloc; called first thing in ``flextool/__init__.py``.

    The runtime path does what the environment path would have done in a process that imported flextool first:
    a ``_RJEM_MALLOC_CONF`` that was set before polars loaded wins as a whole (jemalloc has already read it), and
    otherwise FlexTool's decay values are applied.
    """
    conf = os.environ.get(ENV_VAR)
    if not polars_loaded():
        os.environ.setdefault(ENV_VAR, DEFAULT_CONF)
        return
    if sys.platform == "win32":
        return
    ctl, opt, failure = None, None, None
    try:
        library = runtime_library()
        if library is None or not has_jemalloc(library[0]):
            return
        ctl = Mallctl.from_library(*library)
        opt = {key: ctl.read_ssize(f"opt.{key}") for key in DECAY_KEYS}
    except Exception as exc:  # decided below whether anything needed the call
        ctl, failure = None, exc
    targets, child_conf = plan_decay(conf, opt)
    if child_conf is not None:
        os.environ[ENV_VAR] = child_conf
    if not targets:
        return
    try:
        if ctl is None:
            raise failure or AllocatorUnreachable("mallctl not located")
        apply_decay(ctl, targets)
    except Exception as exc:  # any failure leaves the allocator as it was; the warning says what to do
        warn_once(f"{type(exc).__name__}: {exc}")


def plan_decay(conf: str | None, opt: dict[str, int] | None) -> tuple[dict[str, int], str | None]:
    """(decay values to write at runtime, new ``_RJEM_MALLOC_CONF`` for child processes or None).

    ``opt`` holds the decay values jemalloc read at load (None when mallctl is unreachable).  A leading
    ``dirty_decay_ms:…,muzzy_decay_ms:…`` pair is the one ``import polars`` wrote when jemalloc read exactly those
    values; what follows it was set before polars was imported and wins as a whole.  Any other value was set after
    polars was imported, so jemalloc has not seen it, and its decay keys are written at runtime.
    """
    if not conf:
        return decay_targets(DEFAULT_CONF), DEFAULT_CONF
    prefix = _POLARS_PREFIX.match(conf)
    if prefix is not None and (opt is None or decay_targets(prefix.group(0)) == opt):
        if conf[prefix.end():].strip(","):
            return {}, None
        return decay_targets(DEFAULT_CONF), f"{conf},{DEFAULT_CONF}"
    return decay_targets(conf), None


def polars_loaded() -> bool:
    return any(m == "polars" or m.startswith("_polars_runtime") for m in sys.modules)


def decay_targets(conf: str) -> dict[str, int]:
    """The decay keys of a jemalloc option string; a later occurrence wins, as in jemalloc."""
    out: dict[str, int] = {}
    for item in conf.split(","):
        key, sep, val = item.strip().partition(":")
        if sep and key in DECAY_KEYS:
            try:
                out[key] = int(val)
            except ValueError:
                continue
    return out


def runtime_library() -> tuple[str, str] | None:
    """(path, exported init symbol) of the loaded polars extension library, or None when none is loaded."""
    for name, mod in list(sys.modules.items()):
        if not (name.startswith("_polars_runtime") or name == "polars.polars"):
            continue
        path = getattr(mod, "__file__", None) or ""
        if path.endswith((".so", ".pyd", ".dylib")):
            return path, "PyInit_" + name.rpartition(".")[2]
    return None


def has_jemalloc(path: str) -> bool:
    """jemalloc's message prefix survives stripping, so it tells a jemalloc build from one without."""
    import mmap
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return mm.find(b"<jemalloc>") >= 0


def warn_once(reason: str) -> None:
    if _state["warned"]:
        return
    _state["warned"] = True
    print(REMEDY.format(reason=reason), file=sys.stderr, flush=True)


def elf_function(path: str, names: tuple[str, ...]) -> dict[str, tuple[int, bytes]]:
    """{name: (symbol value, first file bytes of the function)} from the ELF64 ``.symtab`` of ``path``."""
    import mmap
    import struct

    import numpy as np

    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        if mm[:4] != b"\x7fELF" or mm[4] != 2 or mm[5] != 1:
            raise AllocatorUnreachable("not an ELF64 little-endian library")
        shoff, = struct.unpack_from("<Q", mm, 0x28)
        shentsize, shnum = struct.unpack_from("<HH", mm, 0x3A)
        sections = [struct.unpack_from("<IIQQQQIIQQ", mm, shoff + i * shentsize) for i in range(shnum)]
        symtab = next((s for s in sections if s[1] == _SHT_SYMTAB), None)
        if symtab is None:
            raise AllocatorUnreachable("the library has no symbol table")
        sym_off, sym_size, str_index = symtab[4], symtab[5], symtab[6]
        str_off, str_size = sections[str_index][4], sections[str_index][5]
        str_end = str_off + str_size
        name_offsets = {}
        for name in names:
            pos = mm.find(b"\0" + name.encode() + b"\0", str_off, str_end)
            if pos < 0:
                raise AllocatorUnreachable(f"symbol {name} not found")
            name_offsets[name] = pos + 1 - str_off
        dtype = np.dtype([("name", "<u4"), ("info", "u1"), ("other", "u1"), ("shndx", "<u2"),
                          ("value", "<u8"), ("size", "<u8")])
        syms = np.frombuffer(mm, dtype=dtype, count=sym_size // dtype.itemsize, offset=sym_off)
        hits = {name: syms[(syms["name"] == noff) & ((syms["info"] & 0xF) == _STT_FUNC) & (syms["size"] > 0)]
                for name, noff in name_offsets.items()}
        del syms  # copies are kept; the mmap closes only without views into it
        out = {}
        for name, hit in hits.items():
            if len(hit) != 1:
                raise AllocatorUnreachable(f"symbol {name}: {len(hit)} function entries")
            value, shndx = int(hit["value"][0]), int(hit["shndx"][0])
            if not 0 < shndx < shnum:
                raise AllocatorUnreachable(f"symbol {name} is not in a section")
            sec_addr, sec_offset, sec_size = sections[shndx][3], sections[shndx][4], sections[shndx][5]
            n = min(int(hit["size"][0]), _COMPARE_BYTES)
            if not (sec_addr <= value and value + n <= sec_addr + sec_size):
                raise AllocatorUnreachable(f"symbol {name} lies outside its section")
            start = sec_offset + (value - sec_addr)
            out[name] = (value, bytes(mm[start:start + n]))
        return out


class Mallctl:
    """jemalloc's ``int mallctl(const char*, void* oldp, size_t* oldlenp, void* newp, size_t newlen)``."""

    def __init__(self, fn) -> None:
        self._fn = fn

    @classmethod
    def from_library(cls, path: str, init_symbol: str) -> Mallctl:
        import ctypes

        mallctl_symbol = MALLCTL_SYMBOL
        found = elf_function(path, (init_symbol, mallctl_symbol))
        lib = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        init_addr = ctypes.cast(getattr(lib, init_symbol), ctypes.c_void_p).value
        if not init_addr:
            raise AllocatorUnreachable(f"{init_symbol} has no address")
        base = init_addr - found[init_symbol][0]
        for name, (value, file_bytes) in found.items():
            if read_memory(base + value, len(file_bytes)) != file_bytes:
                raise AllocatorUnreachable(f"code of {name} in memory differs from the library file")
        proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                 ctypes.c_void_p, ctypes.c_size_t)
        return cls(proto(base + found[mallctl_symbol][0]))

    def read_ssize(self, name: str) -> int:
        import ctypes
        val, size = ctypes.c_ssize_t(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_ssize_t))
        rc = self._fn(name.encode(), ctypes.byref(val), ctypes.byref(size), None, 0)
        if rc != 0:
            raise AllocatorUnreachable(f"mallctl read {name} returned {rc}")
        return val.value

    def read_unsigned(self, name: str) -> int:
        import ctypes
        val, size = ctypes.c_uint(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_uint))
        rc = self._fn(name.encode(), ctypes.byref(val), ctypes.byref(size), None, 0)
        if rc != 0:
            raise AllocatorUnreachable(f"mallctl read {name} returned {rc}")
        return val.value

    def write_ssize(self, name: str, value: int) -> int:
        import ctypes
        val = ctypes.c_ssize_t(value)
        return self._fn(name.encode(), None, None, ctypes.byref(val), ctypes.sizeof(val))


def read_memory(address: int, n: int) -> bytes:
    import ctypes
    return ctypes.string_at(address, n)


def apply_decay(ctl: Mallctl, targets: dict[str, int]) -> None:
    """Write the decay values as the new-arena default and into every initialised arena."""
    if all(ctl.read_ssize(f"arenas.{key}") == val for key, val in targets.items()):
        return
    narenas = ctl.read_unsigned("arenas.narenas")
    for key, val in targets.items():
        rc = ctl.write_ssize(f"arenas.{key}", val)
        if rc != 0:
            raise AllocatorUnreachable(f"mallctl write arenas.{key} returned {rc}")
        for i in range(narenas):
            rc = ctl.write_ssize(f"arena.{i}.{key}", val)
            if rc not in (0, _EFAULT):
                raise AllocatorUnreachable(f"mallctl write arena.{i}.{key} returned {rc}")
