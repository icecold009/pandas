import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.errors import Pandas4Warning


class TestSeriesArgsort:
    def test_argsort_axis(self):
        # GH#54257
        ser = pd.Series(range(3))

        msg = "No axis named 2 for object type Series"
        with pytest.raises(ValueError, match=msg):
            ser.argsort(axis=2)

    def test_argsort_numpy(self, datetime_series):
        ser = datetime_series
        res = np.argsort(ser).values
        expected = np.argsort(np.array(ser))
        tm.assert_numpy_array_equal(res, expected)

    def test_argsort_numpy_missing(self):
        data = [0.1, np.nan, 0.2, np.nan, 0.3]
        ser = pd.Series(data)
        result = np.argsort(ser)
        expected = np.argsort(np.array(data))

        tm.assert_numpy_array_equal(result.values, expected)

    def test_argsort(self, datetime_series):
        argsorted = datetime_series.argsort()
        assert issubclass(argsorted.dtype.type, np.integer)

    def test_argsort_dt64(self, unit):
        # GH#2967 (introduced bug in 0.11-dev I think)
        ser = pd.Series(
            [pd.Timestamp(f"201301{i:02d}") for i in range(1, 6)], dtype=f"M8[{unit}]"
        )
        assert ser.dtype == f"datetime64[{unit}]"
        shifted = ser.shift(-1)
        assert shifted.dtype == f"datetime64[{unit}]"
        assert pd.isna(shifted[4])

        result = ser.argsort()
        expected = pd.Series(range(5), dtype=np.intp)
        tm.assert_series_equal(result, expected)

        result = shifted.argsort()
        expected = pd.Series([*list(range(4)), 4], dtype=np.intp)
        tm.assert_series_equal(result, expected)

    def test_argsort_stable(self):
        ser = pd.Series(np.random.default_rng(2).integers(0, 100, size=10000))
        mindexer = ser.argsort(kind="mergesort")
        qindexer = ser.argsort()

        mexpected = np.argsort(ser.values, kind="mergesort")
        qexpected = np.argsort(ser.values, kind="quicksort")

        tm.assert_series_equal(mindexer.astype(np.intp), pd.Series(mexpected))
        tm.assert_series_equal(qindexer.astype(np.intp), pd.Series(qexpected))
        msg = (
            r"ndarray Expected type <class 'numpy\.ndarray'>, "
            r"found <class 'pandas\.Series'> instead"
        )
        with pytest.raises(AssertionError, match=msg):
            tm.assert_numpy_array_equal(qindexer, mindexer)

    def test_argsort_stable_keyword(self):
        # GH#64255
        ser = pd.Series(np.random.default_rng(2).integers(0, 100, size=10000))

        true_indexer = ser.argsort(stable=True)
        false_indexer = ser.argsort(stable=False)

        true_expected = np.argsort(ser.values, kind="stable")
        false_expected = np.argsort(ser.values, kind="quicksort")

        tm.assert_numpy_array_equal(true_indexer.values, true_expected)
        tm.assert_numpy_array_equal(false_indexer.values, false_expected)

    def test_argsort_deprecated_pos_arg_stable(self):
        # GH#64255
        ser = pd.Series(np.random.default_rng(2).integers(0, 100, size=10000))
        depr_msg = (
            "Starting with pandas version 4.0 all arguments of argsort except for the "
            "arguments 'axis', 'kind' and 'order' will be keyword-only."
        )

        with tm.assert_produces_warning(Pandas4Warning, match=depr_msg):
            true_indexer = ser.argsort(0, None, None, True)
        with tm.assert_produces_warning(Pandas4Warning, match=depr_msg):
            false_indexer = ser.argsort(0, None, None, False)

        true_expected = np.argsort(ser.values, kind="stable")
        false_expected = np.argsort(ser.values, kind="quicksort")

        tm.assert_numpy_array_equal(true_indexer.values, true_expected)
        tm.assert_numpy_array_equal(false_indexer.values, false_expected)

    @pytest.mark.parametrize("kind", ["quicksort", "mergesort", "heapsort", "stable"])
    @pytest.mark.parametrize("stable", [False, True])
    def test_argsort_kind_and_stable(self, kind, stable):
        # GH#64255
        ser = pd.Series([2, 1, 2, 1])
        msg = "`kind` and `stable` can't be provided at the same time."

        with tm.assert_produces_warning(Pandas4Warning, match=msg):
            result = ser.argsort(kind=kind, stable=stable)
        expected = ser.argsort(kind=kind)
        tm.assert_series_equal(result, expected)

    def test_argsort_numpy_stable(self):
        # GH#64255
        ser = pd.Series(np.random.default_rng(2).integers(0, 100, size=10000))

        true_indexer = np.argsort(ser, stable=True)
        false_indexer = np.argsort(ser, stable=False)

        true_expected = np.argsort(ser.values, kind="stable")
        false_expected = np.argsort(ser.values, kind="quicksort")

        tm.assert_numpy_array_equal(true_indexer.values, true_expected)
        tm.assert_numpy_array_equal(false_indexer.values, false_expected)

    @pytest.mark.parametrize("kind", ["quicksort", "mergesort"])
    @pytest.mark.parametrize("stable", [False, True])
    def test_argsort_numpy_kind_and_stable(self, kind, stable):
        # GH#64255
        ser = pd.Series([2, 1, 2, 1])
        msg = "`kind` and `stable` can't be provided at the same time."

        with tm.assert_produces_warning(
            Pandas4Warning, check_stacklevel=False, match=msg
        ):
            result = np.argsort(ser, kind=kind, stable=stable)
        expected = ser.argsort(kind=kind)
        tm.assert_series_equal(result, expected)

    def test_argsort_preserve_name(self, datetime_series):
        result = datetime_series.argsort()
        assert result.name == datetime_series.name
