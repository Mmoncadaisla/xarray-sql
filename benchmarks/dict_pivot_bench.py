"""dict_coords pivot benchmark: dense vs dictionary coordinates, per VM size.

Measures, on each VM (synthetic in-memory data — no network, so the CPU
and GIL behavior is isolated):

  A. coordinate-production thread scaling (the GIL question): chunks/s
     producing pivoted coordinate columns dense vs dict at 1..N threads.
  B. end-to-end DuckDB: full-scan AVG and GROUP BY lat over a registered
     dataset, dense vs dict, at prefetch 2 and 8 (prefetch sensitivity
     reveals whether the scan pipeline can use its pool or serializes
     on the GIL).
  C. end-to-end DataFusion via register_dataset: same two queries.

Usage:
    python benchmarks/dict_pivot_bench.py --local
    python benchmarks/dict_pivot_bench.py --vms e2-standard-8,e2-standard-16
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

REGION = "us-central1"

# Chunk shape mirrors upstream PR #217's yardstick; 16 chunks ≈ 100M rows.
CHUNK_SHAPE = (6, 721, 1440)
N_CHUNKS = 16
QUERY_REPS = 3


# --------------------------------------------------------------------------
# Remote side
# --------------------------------------------------------------------------


def _install_src(src_targz: bytes | None) -> str:
    import hashlib

    digest = hashlib.md5(src_targz or b"local").hexdigest()[:10]
    root = f"/tmp/xql_dictpivot_src_{digest}"
    marker = os.path.join(root, "xarray_sql", "df.py")
    if src_targz is not None and not os.path.exists(marker):
        os.makedirs(root, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(src_targz), mode="r:gz") as tf:
            tf.extractall(root)  # noqa: S202 — our own tarball
    return root


def run_bench(src_targz: bytes | None = None) -> dict:
    """The whole benchmark; runs where the VM is."""
    import platform
    import threading

    root = _install_src(src_targz)
    if root not in sys.path:
        sys.path.insert(0, root)

    import numpy as np
    import pyarrow as pa
    import xarray as xr

    import xarray_sql as xql
    from xarray_sql.df import _dict_indices

    nt, nlat, nlon = CHUNK_SHAPE
    total_t = nt * N_CHUNKS
    rng = np.random.default_rng(11)
    ds = xr.Dataset(
        {
            "t2m": (
                ("time", "lat", "lon"),
                rng.random((total_t, nlat, nlon), dtype=np.float32),
            )
        },
        coords={
            "time": np.array(
                [
                    np.datetime64("2020-01-01") + np.timedelta64(i, "h")
                    for i in range(total_t)
                ]
            ),
            "lat": np.linspace(90.0, -90.0, nlat),
            "lon": np.linspace(0.0, 359.75, nlon),
        },
    )
    chunks = {"time": nt}
    rows = total_t * nlat * nlon

    results: dict = {
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
        },
        "shape": {"chunk": CHUNK_SHAPE, "n_chunks": N_CHUNKS, "rows": rows},
    }

    # ---- A. coordinate-production thread scaling --------------------------
    shape = CHUNK_SHAPE
    strides = [int(np.prod(shape[k + 1 :])) for k in range(3)]
    outers = [int(np.prod(shape[:k])) for k in range(3)]
    coords = [
        ds.time.values[:nt].copy(),
        ds.lat.values.copy(),
        ds.lon.values.copy(),
    ]
    for k in range(3):
        _dict_indices(shape, k)  # warm the shared cache

    def dense_once() -> None:
        for k in range(3):
            col = np.repeat(coords[k], strides[k])
            if outers[k] > 1:
                col = np.tile(col, outers[k])
            pa.array(col)

    def dict_once() -> None:
        for k in range(3):
            pa.DictionaryArray.from_arrays(
                _dict_indices(shape, k), pa.array(coords[k])
            )

    def scaling(fn, per_thread: int) -> dict[str, float]:
        out = {}
        for n in (1, 2, 4, 8, 16):
            if n > (os.cpu_count() or 1):
                break
            threads = [
                threading.Thread(
                    target=lambda: [fn() for _ in range(per_thread)]
                )
                for _ in range(n)
            ]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            out[str(n)] = round(
                n * per_thread / (time.perf_counter() - t0), 1
            )
        return out

    results["production_chunks_per_s"] = {
        "dense": scaling(dense_once, 8),
        "dict": scaling(dict_once, 80),
    }

    # ---- B/C. end-to-end engine queries ------------------------------------
    def best(fn) -> float:
        times = []
        for _ in range(QUERY_REPS):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return round(min(times), 3)

    # full_avg projects only the data variable (coordinate production is
    # skipped entirely by projection pushdown — the baseline floor);
    # filter_avg forces `lat` production on every chunk for the exact
    # filter; groupby_time_lat produces two coordinate columns and a
    # wide aggregation, the case-05-like row-production shape.
    queries = {
        "full_avg": "SELECT AVG(t2m) FROM t",
        "filter_avg": "SELECT AVG(t2m) FROM t WHERE lat BETWEEN 0 AND 45",
        "groupby_time_lat": (
            "SELECT time, lat, AVG(t2m) AS m FROM t "
            "GROUP BY time, lat ORDER BY time, lat"
        ),
    }

    import duckdb

    duck: dict = {}
    for prefetch in (2, 8):
        for mode, flag in (("dense", False), ("dict", True)):
            dset = xql.arrow_dataset(
                ds, chunks, prefetch=prefetch, dict_coords=flag
            )
            con = duckdb.connect()
            con.register("t", dset)
            for qname, q in queries.items():
                key = f"{qname}/{mode}/prefetch{prefetch}"
                duck[key] = best(lambda q=q, con=con: con.sql(q).fetchall())
            con.close()
    results["duckdb_s"] = duck

    import datafusion

    dfn: dict = {}
    for mode, flag in (("dense", False), ("dict", True)):
        dset = xql.arrow_dataset(ds, chunks, prefetch=8, dict_coords=flag)
        ctx = datafusion.SessionContext()
        ctx.register_dataset("t", dset)
        for qname, q in queries.items():
            dfn[f"{qname}/{mode}"] = best(
                lambda q=q, ctx=ctx: ctx.sql(q).collect()
            )
    results["datafusion_s"] = dfn

    return results


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _pack_src() -> bytes:
    repo = Path(__file__).resolve().parents[1]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path in sorted((repo / "xarray_sql").rglob("*.py")):
            if "__pycache__" not in path.parts:
                tf.add(path, arcname=str(path.relative_to(repo)))
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--vms", default="e2-standard-8,e2-standard-16")
    ap.add_argument("--out", default="dict_pivot_results.json")
    args = ap.parse_args()

    all_results: dict = {}
    if args.local:
        all_results["local"] = run_bench(None)
    else:
        import coiled

        src = _pack_src()
        for vm in args.vms.split(","):
            vm = vm.strip()
            print(f"[{vm}] provisioning + running...", flush=True)
            fn = coiled.function(
                name=f"dict-pivot-{vm}",
                vm_type=vm,
                region=REGION,
                keepalive="5m",
                idle_timeout="10 minutes",
                spot_policy="on-demand",
                package_sync_ignore=["xarray_sql", "xarray-sql"],
                environ={"PYTHONUNBUFFERED": "1"},
            )(run_bench)
            all_results[vm] = fn(src)
            print(f"[{vm}] done", flush=True)

    print(json.dumps(all_results, indent=2))
    Path(args.out).write_text(json.dumps(all_results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
