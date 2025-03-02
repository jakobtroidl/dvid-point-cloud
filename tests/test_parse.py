"""Tests for the parse module."""

import struct
from typing import Dict, List, Tuple

import numpy as np
import pytest

from dvid_point_cloud.parse import decode_block_coord, parse_rles, rles_to_points


def test_decode_block_coord():
    """Test decoding block coordinates from encoded uint64."""
    # Test case 1: All zeros
    encoded = 0
    z, y, x = decode_block_coord(encoded)
    assert z == 0
    assert y == 0
    assert x == 0
    
    # Test case 2: Only X coordinate
    encoded = 42
    z, y, x = decode_block_coord(encoded)
    assert z == 0
    assert y == 0
    assert x == 42
    
    # Test case 3: Only Y coordinate
    encoded = 42 << 21
    z, y, x = decode_block_coord(encoded)
    assert z == 0
    assert y == 42
    assert x == 0
    
    # Test case 4: Only Z coordinate
    encoded = 42 << 42
    z, y, x = decode_block_coord(encoded)
    assert z == 42
    assert y == 0
    assert x == 0
    
    # Test case 5: All coordinates
    encoded = (5 << 42) | (10 << 21) | 15
    z, y, x = decode_block_coord(encoded)
    assert z == 5
    assert y == 10
    assert x == 15
    
    # Test case 6: Maximum values
    max_value = (2**21) - 1  # 21 bits maximum
    encoded = (max_value << 42) | (max_value << 21) | max_value
    z, y, x = decode_block_coord(encoded)
    assert z == max_value
    assert y == max_value
    assert x == max_value


def test_parse_rles():
    """Test parsing of RLE format binary data."""
    # Create a simple RLE-encoded sparse volume
    payload_descriptor = 0  # Binary sparse volume
    num_dimensions = 3
    dimension_of_run = 0  # X dimension
    reserved_byte = 0
    voxel_count = 0  # Placeholder
    
    runs = [
        (10, 20, 30, 5),   # Run of 5 voxels starting at (10, 20, 30)
        (15, 20, 30, 10),  # Run of 10 voxels starting at (15, 20, 30)
        (0, 0, 0, 3)       # Run of 3 voxels starting at (0, 0, 0)
    ]
    
    # Construct binary data
    binary_data = bytearray()
    binary_data.append(payload_descriptor)
    binary_data.append(num_dimensions)
    binary_data.append(dimension_of_run)
    binary_data.append(reserved_byte)
    binary_data.extend(struct.pack("<I", voxel_count))
    binary_data.extend(struct.pack("<I", len(runs)))
    
    for x, y, z, length in runs:
        binary_data.extend(struct.pack("<i", x))
        binary_data.extend(struct.pack("<i", y))
        binary_data.extend(struct.pack("<i", z))
        binary_data.extend(struct.pack("<i", length))
    
    # Parse the RLEs
    parsed_runs = parse_rles(bytes(binary_data))
    
    # Check that parsed runs match the input runs
    assert len(parsed_runs) == len(runs)
    for i, run in enumerate(runs):
        assert parsed_runs[i] == run


def test_rles_to_points():
    """Test conversion of RLEs to point cloud using sample indices."""
    # Define runs
    runs = [
        (10, 20, 30, 5),   # Run of 5 voxels at indices 0-4
        (15, 20, 30, 10),  # Run of 10 voxels at indices 5-14
        (0, 0, 0, 3)       # Run of 3 voxels at indices 15-17
    ]
    
    # Total voxels across all runs
    total_voxels = 5 + 10 + 3
    
    # Sample every other voxel (indices 0, 2, 4, 6, 8, 10, 12, 14, 16)
    sample_indices = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])
    
    # Convert to points
    points = rles_to_points(runs, total_voxels, sample_indices)
    
    # Expected points:
    # From first run: (10, 20, 30), (12, 20, 30), (14, 20, 30)
    # From second run: (16, 20, 30), (18, 20, 30), (20, 20, 30), (22, 20, 30), (24, 20, 30)
    # From third run: (1, 0, 0)
    expected = np.array([
        [10, 20, 30],  # Run 1, offset 0
        [12, 20, 30],  # Run 1, offset 2
        [14, 20, 30],  # Run 1, offset 4
        [16 + 1, 20, 30],  # Run 2, offset 1 (indices 5-14 correspond to this run)
        [16 + 3, 20, 30],  # Run 2, offset 3
        [16 + 5, 20, 30],  # Run 2, offset 5
        [16 + 7, 20, 30],  # Run 2, offset 7
        [16 + 9, 20, 30],  # Run 2, offset 9
        [0 + 1, 0, 0]   # Run 3, offset 1 (indices 15-17 correspond to this run)
    ])
    
    # Check that points match expected
    np.testing.assert_array_equal(points, expected)
    
    # Test with a single sample
    sample_indices = np.array([7])
    points = rles_to_points(runs, total_voxels, sample_indices)
    
    # Sample index 7 corresponds to the 3rd voxel in the second run
    expected = np.array([[15 + 2, 20, 30]])
    np.testing.assert_array_equal(points, expected)
    
    # Test with empty sample
    sample_indices = np.array([], dtype=np.int64)
    points = rles_to_points(runs, total_voxels, sample_indices)
    
    assert points.shape == (0, 3)
    
    # Test with sample indices out of range
    sample_indices = np.array([total_voxels + 1])
    with pytest.raises(IndexError):
        points = rles_to_points(runs, total_voxels, sample_indices)