from types import EllipsisType
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np

from virtualizarr.manifests.array_api import expand_dims
from virtualizarr.manifests.manifest import ChunkManifest
from virtualizarr.manifests.utils import copy_and_replace_metadata

# indexer with only basic selectors, no new axes or ellipsis
T_BasicIndexer_1d: TypeAlias = int | slice | np.ndarray
# indexer allowing new axes but without ellipses
T_SimpleIndexer_1d: TypeAlias = T_BasicIndexer_1d | None
# general valid indexer representing any possible user input
T_Indexer_1d: TypeAlias = T_SimpleIndexer_1d | EllipsisType
T_Indexer: TypeAlias = T_Indexer_1d | tuple[T_Indexer_1d, ...]


if TYPE_CHECKING:
    from virtualizarr.manifests.array import ManifestArray


def index(marr: "ManifestArray", indexer: T_Indexer) -> "ManifestArray":
    """Index into a ManifestArray"""
    indexer_tuple = check_and_sanitize_indexer_type(indexer)
    indexer_without_ellipsis = check_shape_and_maybe_replace_ellipsis(
        indexer_tuple, marr.ndim
    )
    return apply_indexer(marr, indexer_without_ellipsis)


def check_and_sanitize_indexer_type(key: T_Indexer) -> tuple[T_Indexer_1d, ...]:
    """Check for invalid input types, and narrow the return type to a tuple of valid 1D indexers."""
    if isinstance(key, (int, slice, EllipsisType, np.ndarray)) or key is None:
        indexer = cast(tuple[T_Indexer_1d, ...], (key,))
    elif isinstance(key, tuple):
        for dim_indexer in key:
            if (
                not isinstance(dim_indexer, (int, slice, np.ndarray))
                and dim_indexer is not None
                and dim_indexer is not ...
            ):
                raise TypeError(
                    f"indexer must be of type int, slice, ellipsis, None, or np.ndarray; or a tuple of such types. Got {key}"
                )
        indexer = cast(tuple[T_Indexer_1d, ...], key)
    else:
        raise TypeError(
            f"indexer must be of type int, slice, ellipsis, None, or np.ndarray; or a tuple of such types. Got {key}"
        )
    return indexer


def check_shape_and_maybe_replace_ellipsis(
    indexer: tuple[T_Indexer_1d, ...], arr_ndim: int
) -> tuple[T_SimpleIndexer_1d, ...]:
    """Deal with any ellipses, potentially by expanding the indexer to match the shape of the array."""

    num_single_axis_indexing_expressions = len(
        [ind_1d for ind_1d in indexer if ind_1d is not None and ind_1d is not ...]
    )
    num_ellipses = len([ind_1d for ind_1d in indexer if ind_1d is ...])

    if num_ellipses > 1:
        raise ValueError(
            f"Invalid indexer. Indexers containing multiple Ellipses are invalid, but indexer={indexer} contains {num_ellipses} ellipses"
        )

    bad_shape_error_msg = (
        "Invalid indexer for array. Indexer must contain a number of single-axis indexing expressions less than or equal to the length of the array. "
        f"However indexer={indexer} has {num_single_axis_indexing_expressions} single-axis indexing expressions and array has {arr_ndim} dimensions."
        "\nIf concatenating using xarray, ensure all non-coordinate data variables to be concatenated include the concatenation dimension, "
        "or consider passing `data_vars='minimal'` and `coords='minimal'` to the xarray combining function."
    )

    if num_ellipses == 1:
        if num_single_axis_indexing_expressions > arr_ndim:
            raise ValueError(bad_shape_error_msg)
        else:
            return replace_single_ellipsis(
                indexer, arr_ndim, num_single_axis_indexing_expressions
            )
    else:  # num_ellipses == 0
        if num_single_axis_indexing_expressions != arr_ndim:
            raise ValueError(bad_shape_error_msg)
        else:
            return cast(tuple[T_SimpleIndexer_1d, ...], indexer)


def replace_single_ellipsis(
    indexer: tuple[T_Indexer_1d, ...],
    arr_ndim: int,
    num_single_axis_indexing_expressions: int,
) -> tuple[T_SimpleIndexer_1d, ...]:
    """
    Replace ellipsis with 0 or more slice(None)s until there are ndim single-axis indexing expressions (so ignoring Nones).
    """
    indexer_as_list = list(indexer)

    # find this position before modifying in-place
    position_of_ellipsis = indexer_as_list.index(...)

    # need to remove the ellipsis separate from replacement, because the ellipsis may have been totally superfluous already,
    # and it still needs to be removed even if no further replacement will occur
    indexer_as_list.remove(...)

    # replace ellipsis with the equivalent number of no-op slices
    num_extra_slices_needed = arr_ndim - num_single_axis_indexing_expressions
    new_slices = [slice(None)] * num_extra_slices_needed

    # insert multiple elements into one position
    indexer_as_list[position_of_ellipsis:position_of_ellipsis] = new_slices

    return cast(tuple[T_SimpleIndexer_1d, ...], tuple(indexer_as_list))


def apply_indexer(
    marr: "ManifestArray", indexer: tuple[T_SimpleIndexer_1d, ...]
) -> "ManifestArray":
    """
    Apply the simplified indexer to all dimensions of the array.

    Iterates over the axes and applies each indexer one-by-one.
    Encountering a None means inserting a new axis at that position.
    """
    # handles composition of subsetting and expanding dims by following the two-step approach described in https://github.com/data-apis/array-api/pull/408#issuecomment-1091056873

    indexer_without_newaxes: tuple[T_BasicIndexer_1d, ...] = tuple(
        [ind_1d for ind_1d in indexer if ind_1d is not None]
    )
    output_arr = apply_selection(marr, indexer_without_newaxes)

    for position, axis_indexer in enumerate(indexer):
        if axis_indexer is None:
            output_arr = expand_dims(output_arr, axis=position)

    return output_arr


def apply_selection(
    marr: "ManifestArray", indexer_without_newaxes: tuple[T_BasicIndexer_1d, ...]
) -> "ManifestArray":
    """Applies indexes to subset along each dimension."""

    # at this point there should be no ellipsis, no Nones, and one 1D indexer for each axis.
    assert len(indexer_without_newaxes) == marr.ndim

    # Track dimensions that will be dropped by integer indexing
    axes_to_drop = []
    new_shape = list(marr.shape)
    new_chunk_shape = list(marr.chunks)
    chunk_indexers = []

    # Process each dimension
    for axis, (length, chunk_size, indexer_1d) in enumerate(
        zip(marr.shape, marr.chunks, indexer_without_newaxes)
    ):
        chunk_indexer, new_dim_size, drop_axis = process_indexer_for_axis(
            indexer_1d, length, chunk_size, axis
        )
        chunk_indexers.append(chunk_indexer)
        
        if drop_axis:
            axes_to_drop.append(axis)
        else:
            new_shape[axis] = new_dim_size

    # Apply chunk indexing to the manifest
    indexed_paths = marr.manifest._paths[tuple(chunk_indexers)]
    indexed_offsets = marr.manifest._offsets[tuple(chunk_indexers)]
    indexed_lengths = marr.manifest._lengths[tuple(chunk_indexers)]
    
    # Create new manifest with indexed chunks
    new_manifest = ChunkManifest.from_arrays(
        paths=indexed_paths,
        offsets=indexed_offsets,
        lengths=indexed_lengths,
        validate_paths=False,
    )

    # Remove dropped dimensions from shape and chunk_shape
    for axis_offset, axis in enumerate(axes_to_drop):
        adjusted_axis = axis - axis_offset
        new_shape.pop(adjusted_axis)
        new_chunk_shape.pop(adjusted_axis)

    # Update metadata with new shape (handle scalar arrays correctly)
    new_metadata = copy_and_replace_metadata(
        marr.metadata,
        new_shape=new_shape,
        new_chunks=new_chunk_shape,
    )

    # Import here to avoid circular dependency
    from virtualizarr.manifests.array import ManifestArray

    return ManifestArray(chunkmanifest=new_manifest, metadata=new_metadata)


def process_indexer_for_axis(
    indexer_1d: T_BasicIndexer_1d, length: int, chunk_size: int, axis: int
) -> tuple[int | slice, int, bool]:
    """
    Process a single-axis indexer and convert it to chunk-grid indexing.

    Parameters
    ----------
    indexer_1d : int | slice | np.ndarray
        The indexer for this axis
    length : int
        The length of this axis
    chunk_size : int
        The chunk size along this axis
    axis : int
        The axis number (for error messages)

    Returns
    -------
    tuple[int | slice, int, bool]
        - chunk_indexer: The indexer to apply to the chunk grid
        - new_length: The new length of this dimension
        - drop_axis: Whether this dimension should be dropped (integer indexing)
    """
    if isinstance(indexer_1d, slice):
        # Normalize the slice
        start, stop, step = indexer_1d.indices(length)
        
        # Check if it's a no-op
        if start == 0 and stop == length and step == 1:
            return slice(None), length, False
        
        # Check if slice is chunk-aligned
        if start % chunk_size != 0:
            raise ValueError(
                f"Slice start {start} is not aligned with chunk size {chunk_size} on axis {axis}. "
                "Only chunk-aligned slicing is supported."
            )
        if stop % chunk_size != 0 and stop != length:
            raise ValueError(
                f"Slice stop {stop} is not aligned with chunk size {chunk_size} on axis {axis}. "
                "Only chunk-aligned slicing is supported."
            )
        if step != 1:
            if step % 1 != 0:
                raise ValueError(
                    f"Slice step {step} must be an integer."
                )
            # For now, only support step=1 for chunk-aligned slicing
            # Could support other steps if they're multiples of chunk_size
            raise NotImplementedError(
                f"Slice step {step} is not supported. Only step=1 is currently supported for chunk-aligned slicing."
            )
        
        # Convert array indices to chunk indices
        chunk_start = start // chunk_size
        chunk_stop = (stop + chunk_size - 1) // chunk_size  # Round up
        
        # Create chunk indexer
        chunk_indexer = slice(chunk_start, chunk_stop, 1)
        new_length = stop - start
        
        return chunk_indexer, new_length, False
        
    elif isinstance(indexer_1d, int):
        # Handle negative indexing
        if indexer_1d < 0:
            indexer_1d = length + indexer_1d
        
        # Check bounds
        if indexer_1d < 0 or indexer_1d >= length:
            raise IndexError(
                f"Index {indexer_1d} is out of bounds for axis {axis} with size {length}"
            )
        
        # Check if integer index is chunk-aligned (selects start of a chunk)
        if indexer_1d % chunk_size != 0:
            raise ValueError(
                f"Index {indexer_1d} is not aligned with chunk size {chunk_size} on axis {axis}. "
                "Only chunk-aligned indexing is supported."
            )
        
        # Convert array index to chunk index
        chunk_index = indexer_1d // chunk_size
        
        # For chunk-aligned indexing, integer indexing selects a whole chunk, so dimension is NOT dropped
        # Instead, we select a slice of one chunk
        chunk_indexer = slice(chunk_index, chunk_index + 1, 1)
        
        return chunk_indexer, chunk_size, False
        
    elif isinstance(indexer_1d, np.ndarray):
        raise NotImplementedError(
            f"Unsupported indexer. So-called 'fancy indexing' via numpy arrays is not supported."
        )
    else:
        raise TypeError(f"Invalid indexer type: {type(indexer_1d)}")


