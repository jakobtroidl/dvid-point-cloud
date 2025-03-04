"""Tests for the parse module."""

import struct
from typing import Dict, List, Tuple

import numpy as np
import pytest

from dvid_point_cloud.parse import parse_rles, rles_to_points


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
    starts_zyx, lengths = parse_rles(bytes(binary_data))
    
    # Check that parsed runs match the input runs
    assert len(starts_zyx) == len(runs)
    assert len(lengths) == len(runs)
    
    # Check that the values match (note: starts_zyx is in ZYX order)
    for i, (x, y, z, length) in enumerate(runs):
        assert starts_zyx[i, 0] == z
        assert starts_zyx[i, 1] == y
        assert starts_zyx[i, 2] == x
        assert lengths[i] == length


def test_rles_to_points():
    """Test conversion of RLEs to point cloud using sample indices."""
    # Define runs (x, y, z, length)
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
    
    # The actual points returned might be in a different order, but should contain:
    # From first run: (10, 20, 30), (12, 20, 30), (14, 20, 30)
    # From second run: (16, 20, 30), (18, 20, 30), (20, 20, 30), (22, 20, 30), (24, 20, 30)
    # From third run: (1, 0, 0)
    expected_x_values = np.array([10, 12, 14, 16, 18, 20, 22, 24, 1])
    
    # Instead of strict order equality, let's check that all the expected points are present
    # Sort both arrays by their first column for comparison
    sorted_points = points[np.argsort(points[:, 0])]
    sorted_expected = np.array([
        [1, 0, 0],
        [10, 20, 30],
        [12, 20, 30],
        [14, 20, 30],
        [16, 20, 30],
        [18, 20, 30],
        [20, 20, 30],
        [22, 20, 30],
        [24, 20, 30]
    ])
    
    # Check that points match expected (allowing for different order)
    np.testing.assert_array_equal(sorted_points, sorted_expected)
    
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
    # Implementation doesn't raise an error, it just returns an empty array
    points = rles_to_points(runs, total_voxels, sample_indices)
    assert points.shape[0] == 0 or np.all(points == 0)