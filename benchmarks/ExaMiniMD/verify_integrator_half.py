#!/usr/bin/env python3
"""Check initial_integrate_half / final_integrate_half nullify state (bench / pkdb alternate workunits).

Full kernels still match a NumPy reference. Alternate workunits set v (and x for initial) to zero.

  cd eval/benchmarks/ExaMiniMD && python3 verify_integrator_half.py

Uses PK_EXEC_SPACE=Serial by default (override if you want).

Exit 0 if checks pass; else 1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(_SRC))

os.environ.setdefault("PK_EXEC_SPACE", "Serial")
os.environ.setdefault("PK_EXA_SPACE", "Serial")

import pykokkos as pk  # noqa: E402

from integrator_nve import (  # noqa: E402
    final_integrate,
    final_integrate_half,
    initial_integrate,
    initial_integrate_half,
)


def _view2d(a: np.ndarray) -> pk.View:
    n, m = int(a.shape[0]), int(a.shape[1])
    v = pk.View([n, m], pk.double)
    for i in range(n):
        for j in range(m):
            v[i][j] = float(a[i, j])
    return v


def _view1d_int(a: np.ndarray) -> pk.View:
    n = int(a.shape[0])
    v = pk.View([n], pk.int32)
    for i in range(n):
        v[i] = int(a[i])
    return v


def _view_mass(a: np.ndarray) -> pk.View:
    n = int(a.shape[0])
    v = pk.View([n], pk.double)
    for i in range(n):
        v[i] = float(a[i])
    return v


def _to_np2(v: pk.View, n: int, m: int) -> np.ndarray:
    out = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            out[i, j] = float(v[i][j])
    return out


def _ref_initial_full(v0, x0, f, typ, mass, dtf, dtv):
    n = v0.shape[0]
    v = v0.copy()
    x = x0.copy()
    for i in range(n):
        dtfm = dtf / mass[typ[i]]
        v[i] += dtfm * f[i]
        x[i] += dtv * v[i]
    return v, x


def _ref_final_full(v0, f, typ, mass, dtf):
    n = v0.shape[0]
    v = v0.copy()
    for i in range(n):
        dtfm = dtf / mass[typ[i]]
        v[i] += dtfm * f[i]
    return v


def main() -> int:
    n = 32
    dtf = 0.01
    dtv = 0.02
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal((n, 3)).astype(np.float64)
    x0 = rng.standard_normal((n, 3)).astype(np.float64)
    f = rng.standard_normal((n, 3)).astype(np.float64)
    typ = np.zeros(n, dtype=np.int32)
    mass = np.array([2.0], dtype=np.float64)

    tol = 1e-9
    z2 = np.zeros((n, 3), dtype=np.float64)

    # --- initial alternate (null) ---
    v_pk = _view2d(v0.copy())
    x_pk = _view2d(x0.copy())
    ff = _view2d(f)
    tt = _view1d_int(typ)
    mm = _view_mass(mass)
    pk.parallel_for(
        "v_init_half",
        n,
        initial_integrate_half,
        x=x_pk,
        v=v_pk,
        f=ff,
        type=tt,
        mass=mm,
        dtf=dtf,
        dtv=dtv,
    )
    err_v = float(np.max(np.abs(_to_np2(v_pk, n, 3) - z2)))
    err_x = float(np.max(np.abs(_to_np2(x_pk, n, 3) - z2)))
    print(f"initial_integrate_half: max|v|={err_v:.3e} max|x|={err_x:.3e} (expect 0)")

    # --- initial full vs ref ---
    v_full_np, _ = _ref_initial_full(v0, x0, f, typ, mass, dtf, dtv)
    v_pk2 = _view2d(v0.copy())
    x_pk2 = _view2d(x0.copy())
    ff2 = _view2d(f)
    tt2 = _view1d_int(typ)
    mm2 = _view_mass(mass)
    pk.parallel_for(
        "v_init_full", n, initial_integrate, x=x_pk2, v=v_pk2, f=ff2, type=tt2, mass=mm2, dtf=dtf, dtv=dtv
    )
    dv_full = v_full_np - v0
    dv_pk_full = _to_np2(v_pk2, n, 3) - v0
    err_scale = float(np.max(np.abs(dv_pk_full - dv_full)))
    print(f"initial_integrate (full) matches ref: max|dv-dv_ref|={err_scale:.3e}")

    # --- final alternate (null) ---
    v_pk3 = _view2d(v0.copy())
    ff3 = _view2d(f)
    tt3 = _view1d_int(typ)
    mm3 = _view_mass(mass)
    pk.parallel_for("v_fin_half", n, final_integrate_half, v=v_pk3, f=ff3, type=tt3, mass=mm3, dtf=dtf)
    err_vf = float(np.max(np.abs(_to_np2(v_pk3, n, 3) - z2)))
    print(f"final_integrate_half: max|v|={err_vf:.3e} (expect 0)")

    v_np_ff = _ref_final_full(v0, f, typ, mass, dtf)
    v_pk4 = _view2d(v0.copy())
    ff4 = _view2d(f)
    tt4 = _view1d_int(typ)
    mm4 = _view_mass(mass)
    pk.parallel_for("v_fin_full", n, final_integrate, v=v_pk4, f=ff4, type=tt4, mass=mm4, dtf=dtf)
    err_ff = float(np.max(np.abs(_to_np2(v_pk4, n, 3) - v_np_ff)))
    print(f"final_integrate (full) matches ref: max|v-v_ref|={err_ff:.3e}")

    ok = max(err_v, err_x, err_scale, err_vf, err_ff) < tol
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
