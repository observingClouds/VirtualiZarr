import pytest
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import ZarrParser
from obstore.store import LocalStore
from virtualizarr.registry import ObjectStoreRegistry
from pathlib import Path


@pytest.fixture
def vds():
    zarr_store = str('/Users/haukeschulz/Documents/GitHub/mllam-data-prep/example.danra.zarr')
    store = LocalStore(prefix=zarr_store)
    registry = ObjectStoreRegistry({f"file://{zarr_store}": store})
    parser = ZarrParser()
    return open_virtual_dataset(url=zarr_store, registry=registry, parser=parser)


def test_coord_selection(vds):
    assert len(vds.isel(state_feature=slice(1,2)).state_feature) == 1
    assert len(vds.isel(state_feature=slice(1,2)).state_feature_long_name) == 1