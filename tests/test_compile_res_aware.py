"""should_compile resolution awareness — the gate must grow with megapixels WITHOUT moving at all
at the 0.25 MP default.

Why this exists: a real 0.98 MP INT8+compile run on a 32 GB card OOM'd on the first backward at
30.8 GB reserved, because the compile gate was a flat 0.25 MP measurement (20 GB). The fix adds a
resolution term that is algebraically zero at 0.25 MP — these tests pin both halves: the term
engages above 0.25, and the extensively-validated default behaviour is byte-identical.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from fizgig.utils.capabilities import (Capabilities, should_compile,          # noqa: E402
                                       _INT8_COMPILE_PEAK_GB, _NF4_COMPILE_PEAK_GB,
                                       _COMPILE_GB_PER_MP, _HEADROOM_GB)

FAILS = []


def ck(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


CAPS = Capabilities(has_cuda=True, device_name="test", sm=(12, 0),
                    vram_gb=32.0, vram_free_gb=30.0,
                    fp8_matmul=True, int8_matmul_train=True, bitsandbytes=True)
LONG = 100_000  # steps — far past every payback threshold, so only the VRAM gates decide


def sc(kind, vram, mp=None, steps=LONG, swap=0):
    kwargs = {} if mp is None else {"mp": mp}
    return should_compile(steps, kind == "nf4", "bf16" if kind == "int8" else "",
                          swap, vram_gb=vram, caps=CAPS, **kwargs)


# --- 1. the 0.25 MP default is byte-identical to the pre-change behaviour -----------------
# Goldens written against the shipped logic: flat INT8 gate at 21.5 GB, no NF4 gate at all.
for vram, want in ((30.0, True), (21.5, True), (21.4, False), (10.0, False)):
    ok, why = sc("int8", vram)
    ck(f"int8 @0.25MP default, {vram} GB free -> {'compile' if want else 'decline'}",
       ok == want, why)
ok, why = sc("int8", 21.4)
ck("  the decline reason is the flat-peak wording (no MP mention)",
   "peaks near 20 GB and" in why and "MP" not in why, why)
for vram in (30.0, 14.0, 8.0):
    ok, why = sc("nf4", vram)
    ck(f"nf4 @0.25MP default, {vram} GB free -> compile (no VRAM gate, as before)", ok, why)

# mp omitted and mp=0.25 must be the same decision AND the same words — any caller that
# doesn't pass mp keeps exactly the old behaviour.
for kind in ("int8", "nf4"):
    for vram in (30.0, 21.4, 14.0):
        ck(f"{kind} {vram} GB: mp omitted == mp=0.25",
           sc(kind, vram) == sc(kind, vram, mp=0.25))

# --- 2. above 0.25 MP the gate grows ------------------------------------------------------
# The run that motivated this: 0.98 MP, ~30 GB free, INT8 -> must now decline.
ok, why = sc("int8", 30.0, mp=0.98)
ck("int8 @0.98MP, 30 GB free -> DECLINED (the OOM run)", not ok, why)
ck("  reason names the resolution and the fix",
   "0.98 MP" in why and "Target Megapixels" in why, why)
ok, why = sc("int8", 32.5, mp=0.98)
ck("int8 @0.98MP with enough free VRAM still compiles", ok, why)

# NF4 gains a gate ONLY above 0.25 MP (extrapolated slope): a 16 GB card at 1 MP declines,
# a big card at 1 MP still compiles.
ok, why = sc("nf4", 14.0, mp=1.0)
ck("nf4 @1.0MP, 14 GB free -> DECLINED", not ok, why)
ok, why = sc("nf4", 30.0, mp=1.0)
ck("nf4 @1.0MP, 30 GB free -> still compiles", ok, why)

# The thresholds are exactly base + slope*(mp-0.25) + headroom — no hidden constants.
_mp = 0.75
_gate = _INT8_COMPILE_PEAK_GB + _COMPILE_GB_PER_MP * (_mp - 0.25) + _HEADROOM_GB
ck("int8 gate sits exactly at the formula",
   sc("int8", _gate + 0.01, mp=_mp)[0] and not sc("int8", _gate - 0.01, mp=_mp)[0])
_gate = _NF4_COMPILE_PEAK_GB + _COMPILE_GB_PER_MP * (_mp - 0.25) + _HEADROOM_GB
ck("nf4 gate sits exactly at the formula",
   sc("nf4", _gate + 0.01, mp=_mp)[0] and not sc("nf4", _gate - 0.01, mp=_mp)[0])

# --- 3. the earlier gates still fire first, untouched by mp -------------------------------
ok, why = sc("int8", 30.0, mp=2.0, swap=8)
ck("block swap still declines before any VRAM math", not ok and "block swap" in why, why)
ok, why = should_compile(LONG, False, "", 0, vram_gb=30.0, caps=CAPS, mp=2.0)
ck("fp8/bf16 still declines regardless of mp", not ok and "quantised paths" in why, why)
ok, why = sc("int8", 30.0, mp=0.25, steps=100)
ck("short runs still decline on payback at the default", not ok and "too short" in why, why)

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
