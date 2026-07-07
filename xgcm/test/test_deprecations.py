import pytest

import xgcm
from xgcm.test.datasets import datasets

pytestmark = pytest.mark.filterwarnings("error")


def test_periodic_removed():
    # `periodic` was removed in v1.0.0 (#746); passing it now raises an
    # informative error pointing at `padding`.
    ds = datasets["2d_left"]
    with pytest.raises(
        ValueError,
        match="The `periodic` argument has been removed",
    ):
        xgcm.Grid(
            ds,
            coords={
                "X": {"left": "XG", "center": "XC"},
                "Y": {"left": "YG", "center": "YC"},
            },
            autoparse_metadata=False,
            periodic=True,
        )


def test_padding_deprecation():
    ds = datasets["2d_left"]
    with pytest.raises(
        ValueError, match="Argument 'boundary' has been renamed to 'padding'"
    ):
        xgcm.Grid(
            ds,
            coords={
                "X": {"left": "XG", "center": "XC"},
                "Y": {"left": "YG", "center": "YC"},
            },
            autoparse_metadata=False,
            boundary="periodic",
        )
