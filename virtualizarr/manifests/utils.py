import re
from typing import TYPE_CHECKING, Any, Dict, Iterable, Literal, Optional, Union

import numpy as np
from zarr import Array
from zarr.core.chunk_key_encodings import ChunkKeyEncodingLike
from zarr.core.metadata.v3 import (
    ArrayV3Metadata,
    parse_dimension_names,
    parse_shapelike,
)
from zarr.dtype import parse_data_type

from virtualizarr.codecs import convert_to_codec_pipeline, get_codecs

if TYPE_CHECKING:
    from .array import ManifestArray


def parse_manifest_index(
    key: str, chunk_key_encoding: Literal[".", "/"] = ".", expand_pattern: bool = False
) -> tuple[int, ...]:
    """
    Extracts the chunk index from a `key` (a.k.a `node`) that represents a chunk of
    data in a Zarr hierarchy. The returned tuple can be used to index the ndarrays
    containing paths, offsets, and lengths in ManifestArrays.

    Parameters
    ----------
    key
        The key in the Zarr store to parse.
    chunk_key_encoding
        The chunk key separator used in the Zarr store.
    expand_pattern
        Whether to expand the pattern matching to include /c to protect against group structures that look like chunks

    Returns
    -------
    tuple containing chunk indexes.

    Raises
    ------
    ValueError
        Raised if the key does not match the expected node structure for a chunk according the
        [Zarr V3 specification][https://zarr-specs.readthedocs.io/en/latest/v3/chunk-key-encodings/index.html].

    """
    # Keys ending in `/c` are scalar arrays. The paths, offsets, and lengths in a chunk manifest
    # of a scalar array should also be scalar arrays that can be indexed with an empty tuple.
    if key.endswith("/c") or key == "c":
        return ()

    pattern = construct_chunk_pattern(chunk_key_encoding)
    if expand_pattern:
        # Expand pattern to include `/c` to protect against group structures that look like chunk structures
        pattern = rf"(?:^|/)c{chunk_key_encoding}{pattern}"
    # Look for f"/c{chunk_key_encoding"}" followed by digits and more /digits
    match = re.search(pattern, key)
    if not match:
        raise ValueError(
            f"Key {key} with chunk_key_encoding {chunk_key_encoding} did not match the expected pattern for nodes in the Zarr hierarchy."
        )
    chunk_component = (
        match.group().removeprefix("/").removeprefix(f"c{chunk_key_encoding}")
    )
    return tuple(int(ind) for ind in chunk_component.split(chunk_key_encoding))


def construct_chunk_pattern(chunk_key_encoding: Literal[".", "/"]) -> str:
    """
    Produces a pattern for finding a chunk indices from key within a Zarr store using [re.match][] or [re.search][].

    Parameters
    ----------
    chunk_key_encoding
        The chunk key separator used in the Zarr store.

    Returns
    -------
    String representation of regular expression for a chunk key index
    """

    integer_pattern = r"([1-9]+\d*|0)"  # matches 0 or an unsigned integer that does not begin with zero
    separator = (
        rf"\{chunk_key_encoding}" if chunk_key_encoding == "." else chunk_key_encoding
    )
    pattern = rf"(c|{integer_pattern}+({separator}{integer_pattern})*)$"  # matches the character "c" or 1 integer, optionally followed by more integers each separated by a separator (i.e. a period)
    return pattern


def create_v3_array_metadata(
    shape: tuple[int, ...],
    data_type: np.dtype,
    chunk_shape: tuple[int, ...],
    chunk_key_encoding: ChunkKeyEncodingLike = {"name": "default"},
    fill_value: Any = None,
    codecs: Optional[list[Dict[str, Any]]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    dimension_names: Iterable[str] | None = None,
) -> ArrayV3Metadata:
    """
    Create an ArrayV3Metadata instance with standard configuration.
    This function encapsulates common patterns used across different parsers.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the array
    data_type : np.dtype
        The numpy dtype of the array
    chunk_shape : tuple[int, ...]
        The shape of each chunk
    chunk_key_encoding : ChunkKeyEncodingLike
        The mapping from chunk grid cell coordinates to keys.
    fill_value : Any, optional
        The fill value for the array
    codecs : list[Dict[str, Any]], optional
        List of codec configurations
    attributes : Dict[str, Any], optional
        Additional attributes for the array
    dimension_names : tuple[str], optional
        Names of the dimensions

    Returns
    -------
    ArrayV3Metadata
        A configured ArrayV3Metadata instance with standard defaults
    """
    zdtype = parse_data_type(data_type, zarr_format=3)
    return ArrayV3Metadata(
        shape=shape,
        data_type=zdtype,
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": chunk_shape},
        },
        chunk_key_encoding=chunk_key_encoding,
        fill_value=zdtype.default_scalar() if fill_value is None else fill_value,
        codecs=convert_to_codec_pipeline(
            codecs=codecs or [],
            dtype=data_type,
        ),
        attributes=attributes or {},
        dimension_names=dimension_names,
        storage_transformers=None,
    )


def check_same_dtypes(dtypes: list[np.dtype]) -> None:
    """Check all the dtypes are the same"""

    first_dtype, *other_dtypes = dtypes
    for other_dtype in other_dtypes:
        if other_dtype != first_dtype:
            raise ValueError(
                f"Cannot concatenate arrays with inconsistent dtypes: {other_dtype} vs {first_dtype}"
            )


def check_compatible_encodings(encoding1, encoding2):
    for key, value in encoding1.items():
        if key in encoding2:
            if encoding2[key] != value:
                raise ValueError(
                    f"Cannot concatenate arrays with different values for encoding key {key}: {encoding2[key]} != {value}"
                )


def _normalize_codec(codec: Any) -> dict[str, Any]:
    """Return a canonical dict representation of *codec*.

    The pipeline comparison used by ``ManifestArray`` previously relied on
    object equality, which meant that two semantically equivalent blosc
    codecs would compare as different if one was a ``numcodecs`` instance and
    the other was the Zarr-built ``BloscCodec`` wrapper.  This helper uses
    ``virtualizarr.codecs.get_codec_config`` to extract the configuration and
    then strips any ``numcodecs.`` prefix.  The resulting dictionaries are
    comparable by value and are guaranteed to be identical for codecs that
    behave the same even if they come from different registrations.
    """

    from virtualizarr.codecs import get_codec_config, zarr_codec_config_to_v3

    cfg = get_codec_config(codec)
    # ensure we have a v3-style dict with ``name``/``configuration``
    if "id" in cfg:
        cfg = zarr_codec_config_to_v3(cfg)
    # strip ``numcodecs.`` prefix so that ``blosc`` and
    # ``numcodecs.blosc`` compare equal
    name = cfg.get("name")
    if isinstance(name, str) and name.startswith("numcodecs."):
        cfg = {"name": name.split(".", 1)[1], "configuration": cfg.get("configuration", {})}

    # The Blosc codec has a ``typesize`` attribute that is automatically
    # filled by zarr based on the array dtype; numcodecs codecs frequently
    # omit the key entirely.  In practice the value has no semantic effect
    # on the data, and the presence/absence of a default value was causing
    # otherwise identical pipelines to be treated as different.  Drop the
    # key entirely for normalization so that ``Blosc`` codecs are compared
    # only on the meaningful portion of their configuration.
    if cfg.get("name") == "blosc":
        conf = cfg.get("configuration", {})
        # remove any “typesize” entry regardless of its value – the
        # dtype check upstream already guarantees that arrays have a
        # matching element size, so this field never needs to influence
        # equality.
        conf.pop("typesize", None)

    # Convert any enum.Enum values (e.g. BloscCodec attributes) into
    # their underlying values so that the canonical representation
    # is JSON-serializable and consistent regardless of whether the
    # codec originated from ``numcodecs`` or the zarr registry.
    def _simplify(obj: Any) -> Any:
        from enum import Enum

        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: _simplify(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_simplify(v) for v in obj)
        else:
            return obj

    return _simplify(cfg)


def _canonical_pipeline(pipeline: Any) -> tuple[dict[str, Any], ...]:
    """Convert a codec pipeline (usually a tuple) into a tuple of normalized dicts."""

    return tuple(_normalize_codec(c) for c in pipeline)


def check_same_codecs(codecs: list[Any]) -> None:
    # ``codecs`` is a list of codec pipelines, e.g. the result of
    # ``[get_codecs(arr) for arr in arrays]``.  Compare pipelines by their
    # canonical dicts rather than relying on object identity.
    first, *others = codecs
    first_canonical = _canonical_pipeline(first)
    for idx, other in enumerate(others, start=1):
        other_canonical = _canonical_pipeline(other)
        if other_canonical != first_canonical:
            raise NotImplementedError(
                "The ManifestArray class cannot concatenate arrays which were stored using different codecs, "
                f"But found codecs {first} vs {other}. "
                f"(canonical: {first_canonical} vs {other_canonical}) "
                "See https://github.com/zarr-developers/zarr-specs/issues/288"
            )


def check_same_chunk_shapes(chunks_list: list[tuple[int, ...]]) -> None:
    """Check all the chunk shapes are the same"""

    first_chunks, *other_chunks_list = chunks_list
    for other_chunks in other_chunks_list:
        if other_chunks != first_chunks:
            raise ValueError(
                f"Cannot concatenate arrays with inconsistent chunk shapes: {other_chunks} vs {first_chunks} ."
                "Requires ZEP003 (Variable-length Chunks)."
            )


def check_same_ndims(ndims: list[int]) -> None:
    first_ndim, *other_ndims = ndims
    for other_ndim in other_ndims:
        if other_ndim != first_ndim:
            raise ValueError(
                f"Cannot concatenate arrays with differing number of dimensions: {first_ndim} vs {other_ndim}"
            )


def check_same_shapes(shapes: list[tuple[int, ...]]) -> None:
    first_shape, *other_shapes = shapes
    for other_shape in other_shapes:
        if other_shape != first_shape:
            raise ValueError(
                f"Cannot concatenate arrays with differing shapes: {first_shape} vs {other_shape}"
            )


def _remove_element_at_position(t: tuple[int, ...], pos: int) -> tuple[int, ...]:
    new_l = list(t)
    new_l.pop(pos)
    return tuple(new_l)


def check_no_partial_chunks_on_concat_axis(
    shapes: list[tuple[int, ...]], chunks: list[tuple[int, ...]], axis: int
):
    """Check that there are no partial chunks along the concatenation axis"""
    # loop over the arrays to be concatenated
    for i, (shape, chunk_shape) in enumerate(zip(shapes, chunks)):
        if shape[axis] % chunk_shape[axis] > 0:
            raise ValueError(
                "Cannot concatenate arrays with partial chunks because only regular chunk grids are currently supported. "
                f"Concat input {i} has array length {shape[axis]} along the concatenation axis which is not "
                f"evenly divisible by chunk length {chunk_shape[axis]}."
            )


def check_same_shapes_except_on_concat_axis(shapes: list[tuple[int, ...]], axis: int):
    """Check that shapes are compatible for concatenation"""

    shapes_without_concat_axis = [
        _remove_element_at_position(shape, axis) for shape in shapes
    ]

    first_shape, *other_shapes = shapes_without_concat_axis
    for other_shape in other_shapes:
        if other_shape != first_shape:
            raise ValueError(
                f"Cannot concatenate arrays with shapes {[shape for shape in shapes]}"
            )


def check_combinable_zarr_arrays(
    arrays: Iterable[Union["ManifestArray", "Array"]],
) -> None:
    """
    The downside of the ManifestArray approach compared to the VirtualZarrArray concatenation proposal is that
    the result must also be a single valid zarr array, implying that the inputs must have the same dtype, codec etc.
    """
    check_same_dtypes([arr.dtype for arr in arrays])

    # Can't combine different codecs in one manifest
    # see https://github.com/zarr-developers/zarr-specs/issues/288
    check_same_codecs([get_codecs(arr) for arr in arrays])

    # Would require variable-length chunks ZEP
    check_same_chunk_shapes([arr.chunks for arr in arrays])


def check_compatible_arrays(
    ma: "ManifestArray", existing_array: "Array", append_axis: int
):
    check_combinable_zarr_arrays([ma, existing_array])
    check_same_ndims([ma.ndim, existing_array.ndim])
    arr_shapes = [ma.shape, existing_array.shape]
    check_same_shapes_except_on_concat_axis(arr_shapes, append_axis)


def copy_and_replace_metadata(
    old_metadata: ArrayV3Metadata,
    new_shape: list[int] | None = None,
    new_chunks: list[int] | None = None,
    new_dimension_names: Iterable[str] | None | Literal["default"] = "default",
    new_attributes: dict | None = None,
) -> ArrayV3Metadata:
    """
    Update metadata to reflect a new shape and/or chunk shape.
    """
    # TODO this should really be upstreamed into zarr-python

    metadata_copy = old_metadata.to_dict().copy()

    if new_shape is not None:
        metadata_copy["shape"] = parse_shapelike(new_shape)  # type: ignore[assignment]
    if new_chunks is not None:
        metadata_copy["chunk_grid"] = {
            "name": "regular",
            "configuration": {"chunk_shape": tuple(new_chunks)},
        }
    if new_dimension_names != "default":
        # need the option to use the literal string "default" as a sentinel value because None is a valid choice for zarr dimension_names
        metadata_copy["dimension_names"] = parse_dimension_names(new_dimension_names)
    if new_attributes is not None:
        metadata_copy["attributes"] = new_attributes

    # ArrayV3Metadata.from_dict removes extra keys zarr_format and node_type
    new_metadata = ArrayV3Metadata.from_dict(metadata_copy)
    return new_metadata
