"""dict_coords=True must be indistinguishable from dense, only cheaper.

Every test here runs the same operation against a dense and a
dictionary-encoded registration of the same Dataset and asserts equal
results. int32 keys per the overflow analysis on upstream PR #217.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import xarray as xr

import xarray_sql as xql
from xarray_sql.df import _dict_indices, _parse_schema

duckdb = pytest.importorskip("duckdb")


def _grid(nt=8, nlat=15, nlon=20) -> xr.Dataset:
    rng = np.random.default_rng(7)
    return xr.Dataset(
        {
            "t2m": (
                ("time", "lat", "lon"),
                rng.random((nt, nlat, nlon)).astype(np.float32),
            ),
            "msl": (
                ("time", "lat", "lon"),
                rng.random((nt, nlat, nlon)),
            ),
        },
        coords={
            "time": np.array(
                [np.datetime64("2020-01-01") + np.timedelta64(i, "h") for i in range(nt)]
            ),
            "lat": np.linspace(60.0, 40.0, nlat),
            "lon": np.linspace(-10.0, 10.0, nlon),
        },
    ).chunk({"time": 3, "lat": 8})


def test_schema_marks_dim_coords_dictionary():
    ds = _grid()
    schema = _parse_schema(ds, dict_coords=True)
    for dim in ("time", "lat", "lon"):
        t = schema.field(dim).type
        assert pa.types.is_dictionary(t)
        assert t.index_type == pa.int32()
    assert schema.field("t2m").type == pa.float32()


def test_dense_dims_stay_dense():
    ds = _grid()
    schema = _parse_schema(ds, dict_coords=True, dense_dims=("lat", "lon"))
    assert pa.types.is_dictionary(schema.field("time").type)
    assert schema.field("lat").type == pa.float64()


def test_dict_indices_cached_and_correct():
    shape = (3, 4, 5)
    a = _dict_indices(shape, 1)
    assert a is _dict_indices(shape, 1)  # cache hit returns same object
    expect = np.tile(np.repeat(np.arange(4), 5), 3)
    np.testing.assert_array_equal(a.to_numpy(), expect)


def _tables(ds, **kw):
    dense = xql.arrow_dataset(ds, **kw)
    encoded = xql.arrow_dataset(ds, dict_coords=True, **kw)
    return dense, encoded


def test_to_table_values_match():
    ds = _grid()
    dense, encoded = _tables(ds)
    a = dense.to_table()
    b = encoded.to_table()
    assert b.schema.field("lat").type == pa.dictionary(pa.int32(), pa.float64())
    # decode and compare full contents
    decoded = pa.table(
        {
            n: (
                b[n].combine_chunks().dictionary_decode()
                if pa.types.is_dictionary(b.schema.field(n).type)
                else b[n]
            )
            for n in b.schema.names
        }
    )
    assert decoded.cast(a.schema).equals(a)


@pytest.mark.parametrize(
    "flt",
    [
        pc.field("lat") > 50.0,
        pc.field("lat") == 60.0,
        pc.field("lat").isin([60.0, 40.0]),
        (pc.field("lat") > 45.0) & (pc.field("lon") < 0.0),
    ],
)
def test_filters_match_dense(flt):
    ds = _grid()
    dense, encoded = _tables(ds)
    assert encoded.to_table(filter=flt).num_rows == dense.to_table(filter=flt).num_rows
    assert encoded.count_rows(filter=flt) == dense.count_rows(filter=flt)


def test_pruning_still_prunes():
    ds = _grid()
    seen: list = []
    encoded = xql.arrow_dataset(ds, dict_coords=True)
    encoded._iteration_callback = lambda block, cols: seen.append(block)
    # time is chunked 3/3/2; a first-hour filter must touch only chunk 0
    flt = pc.field("time") == ds.time.values[0]
    encoded.to_table(filter=flt)
    assert seen and all(b["time"] == slice(0, 3) for b in seen)


def test_duckdb_groupby_and_filter_match_dense():
    ds = _grid()
    con = duckdb.connect()
    dense, encoded = _tables(ds)
    con.register("d", dense)
    con.register("e", encoded)
    q = (
        "SELECT lat, AVG(t2m) AS m FROM {t} "
        "WHERE lon BETWEEN -5 AND 5 GROUP BY lat ORDER BY lat"
    )
    assert con.sql(q.format(t="d")).fetchall() == con.sql(q.format(t="e")).fetchall()


def test_datafusion_register_dataset_matches_dense():
    datafusion = pytest.importorskip("datafusion")
    ds = _grid()
    ctx = datafusion.SessionContext()
    dense, encoded = _tables(ds)
    ctx.register_dataset("d", dense)
    ctx.register_dataset("e", encoded)
    q = (
        "SELECT lat, AVG(t2m) AS m FROM {t} "
        "WHERE lon > 0 GROUP BY lat ORDER BY lat"
    )
    a = ctx.sql(q.format(t="d")).collect()
    b = ctx.sql(q.format(t="e")).collect()
    ta, tb = pa.Table.from_batches(a), pa.Table.from_batches(b)
    lat_b = tb["lat"]
    if pa.types.is_dictionary(lat_b.type):
        lat_b = lat_b.combine_chunks().dictionary_decode()
    np.testing.assert_array_equal(ta["lat"].to_numpy(), lat_b.to_numpy())
    np.testing.assert_allclose(ta["m"].to_numpy(), tb["m"].to_numpy())


def test_roundtrip_to_dataset_matches_dense():
    ds = _grid()
    con = duckdb.connect()
    dense, encoded = _tables(ds)
    con.register("d", dense)
    con.register("e", encoded)
    q = (
        "SELECT time, lat, lon, t2m FROM {t} "
        "WHERE lat > 50 ORDER BY time, lat, lon"
    )
    out_d = xql.to_dataset(con.sql(q.format(t="d")), template=ds)
    out_e = xql.to_dataset(con.sql(q.format(t="e")), template=ds)
    xr.testing.assert_identical(out_d, out_e)


def test_geometry_dims_forced_dense():
    ds = _grid()
    encoded = xql.arrow_dataset(ds, dict_coords=True, geometry=("lon", "lat"))
    for dim in ("lat", "lon"):
        assert not pa.types.is_dictionary(encoded.schema.field(dim).type)
    assert pa.types.is_dictionary(encoded.schema.field("time").type)
    # geometry column still materializes
    t = encoded.to_table(columns=["lat", "lon", "geometry"])
    assert t.num_rows == 8 * 15 * 20
