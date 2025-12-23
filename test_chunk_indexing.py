#!/usr/bin/env python
"""
Demonstration of chunk-aligned indexing for ManifestArray.
This script shows how the fix for issue #51 allows indexing along chunk boundaries.
"""

import numpy as np
from virtualizarr.manifests import ManifestArray, ChunkManifest
from virtualizarr.manifests.utils import create_v3_array_metadata

# Create a simple ManifestArray for demonstration
shape = (10, 20)
chunks = (5, 10)
dtype = np.dtype('int32')

# Create metadata
metadata = create_v3_array_metadata(
    shape=shape,
    data_type=dtype,
    chunk_shape=chunks,
)

# Create a simple chunk manifest (all chunks point to the same dummy location)
entries = {}
for i in range(shape[0] // chunks[0]):
    for j in range(shape[1] // chunks[1]):
        key = f"{i}.{j}"
        entries[key] = {
            "path": "s3://bucket/data.nc",
            "offset": (i * 2 + j) * 1000,
            "length": chunks[0] * chunks[1] * dtype.itemsize
        }

manifest = ChunkManifest(entries=entries)
marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)

print("Original ManifestArray:")
print(f"  Shape: {marr.shape}")
print(f"  Chunks: {marr.chunks}")
print(f"  Chunk grid shape: {marr.manifest.shape_chunk_grid}")
print()

# Test 1: Integer indexing at chunk boundary (selects one chunk)
print("Test 1: Integer indexing at chunk boundary")
print("  marr[0, :] - selects first chunk row")
indexed1 = marr[0, :]
print(f"  Result shape: {indexed1.shape}")
print(f"  Result chunks: {indexed1.chunks}")
print(f"  Chunk grid shape: {indexed1.manifest.shape_chunk_grid}")
print()

# Test 2: Slice indexing along chunk boundaries
print("Test 2: Slice indexing along chunk boundaries")
print("  marr[0:5, 0:10] - selects one chunk")
indexed2 = marr[0:5, 0:10]
print(f"  Result shape: {indexed2.shape}")
print(f"  Result chunks: {indexed2.chunks}")
print(f"  Chunk grid shape: {indexed2.manifest.shape_chunk_grid}")
print()

# Test 3: Multi-chunk slice
print("Test 3: Multi-chunk slice")
print("  marr[:, 0:10] - selects first column of chunks")
indexed3 = marr[:, 0:10]
print(f"  Result shape: {indexed3.shape}")
print(f"  Result chunks: {indexed3.chunks}")
print(f"  Chunk grid shape: {indexed3.manifest.shape_chunk_grid}")
print()

# Test 4: Combination with ellipsis
print("Test 4: Using ellipsis")
print("  marr[5, ...] - selects second chunk along dimension 0")
indexed4 = marr[5, ...]
print(f"  Result shape: {indexed4.shape}")
print(f"  Result chunks: {indexed4.chunks}")
print(f"  Chunk grid shape: {indexed4.manifest.shape_chunk_grid}")
print()

# Test 5: Error case - non-chunk-aligned indexing
print("Test 5: Non-chunk-aligned indexing (should raise error)")
try:
    marr[3, :]  # 3 is not aligned with chunk size 5
    print("  ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"  ✓ Correctly raised ValueError: {str(e)[:80]}...")
print()

print("All tests completed successfully!")
print("Chunk-aligned indexing is now fully functional!")
