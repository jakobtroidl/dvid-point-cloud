"""Functions for parsing DVID data formats."""

import logging
import struct
from typing import Dict, List, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def decode_block_coord(encoded_coord: int) -> Tuple[int, int, int]:
    """
    Decode a packed block coordinate into (z, y, x) coordinates.
    
    Args:
        encoded_coord: Block coordinate encoded as a uint64
        
    Returns:
        Tuple of (z, y, x) coordinates (positive integers)
    """
    # DVID block coordinates are assumed to be positive integers
    x = encoded_coord & 0x1FFFFF          # Extract bits 0-20 for X
    y = (encoded_coord >> 21) & 0x1FFFFF  # Extract bits 21-41 for Y
    z = (encoded_coord >> 42) & 0x1FFFFF  # Extract bits 42-62 for Z
    
    return z, y, x


def parse_rles(binary_data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse run-length encoded data from DVID's sparsevol format.
    
    Args:
        binary_data: Binary data in DVID RLE format
        
    Returns:
        Tuple of (starts_zyx, lengths) where:
          - starts_zyx is a numpy array of shape (N, 3) with the ZYX start coordinate of each run
          - lengths is a numpy array of shape (N,) with the length of each run along X axis
    """
    offset = 0
    
    # Parse header
    payload_descriptor = binary_data[offset]
    offset += 1
    
    num_dimensions = binary_data[offset]
    offset += 1
    
    dimension_of_run = binary_data[offset]
    offset += 1
    
    # Skip reserved byte
    offset += 1
    
    # Skip voxel count (uint32)
    offset += 4
    
    # Number of spans
    num_spans = struct.unpack("<I", binary_data[offset:offset+4])[0]
    offset += 4
    
    logger.debug(f"Sparse volume RLE: {num_spans} spans, payload_descriptor={payload_descriptor}")
    
    # Pre-allocate arrays for vectorized processing
    starts_xyz = np.zeros((num_spans, 3), dtype=np.int32)
    lengths = np.zeros(num_spans, dtype=np.int32)
    
    # Parse RLE spans
    for i in range(num_spans):
        x = struct.unpack("<i", binary_data[offset:offset+4])[0]
        offset += 4
        
        y = struct.unpack("<i", binary_data[offset:offset+4])[0]
        offset += 4
        
        z = struct.unpack("<i", binary_data[offset:offset+4])[0]
        offset += 4
        
        run_length = struct.unpack("<i", binary_data[offset:offset+4])[0]
        offset += 4
        
        starts_xyz[i] = [x, y, z]
        lengths[i] = run_length
        
        # Skip payload if present
        if payload_descriptor > 0:
            payload_size = 0  # Calculate based on descriptor and run_length
            offset += payload_size
    
    # Convert from XYZ to ZYX for easier numpy operations
    starts_zyx = starts_xyz[:, ::-1]
    
    return starts_zyx, lengths


def rles_to_points(runs: List[Tuple[int, int, int, int]], total_voxels: int, 
                   sample_indices: np.ndarray) -> np.ndarray:
    """
    Convert runs to randomly sampled points based on sample indices.
    
    Args:
        runs: List of (x, y, z, length) tuples representing runs
        total_voxels: Total number of voxels across all runs
        sample_indices: 1D array of indices to sample (must be sorted)
        
    Returns:
        Numpy array of shape (N, 3) with sampled XYZ coordinates
    """
    # Pre-allocate points array
    num_samples = len(sample_indices)
    points = np.zeros((num_samples, 3), dtype=np.int32)
    
    current_index = 0
    voxel_counter = 0
    sample_idx = 0
    
    # Process each run
    for x, y, z, run_length in runs:
        # If we've sampled all points, we can exit early
        if sample_idx >= num_samples:
            break
            
        # Check if any sample points fall within this run
        run_start = voxel_counter
        run_end = run_start + run_length
        
        # Process all sample points that fall within this run
        while (sample_idx < num_samples and 
               sample_indices[sample_idx] >= run_start and 
               sample_indices[sample_idx] < run_end):
            
            # Calculate the offset within the run
            offset = sample_indices[sample_idx] - run_start
            
            # Store the point (assuming X is the dimension of the run)
            points[sample_idx] = [x + offset, y, z]
            
            # Move to next sample
            sample_idx += 1
            
        # Update the voxel counter
        voxel_counter = run_end
    
    return points


def vectorized_sample_from_rles(starts_zyx: np.ndarray, lengths: np.ndarray, 
                               num_points: int) -> np.ndarray:
    """
    Generate a point cloud sample from run-length encoded data using vectorized operations.
    
    Args:
        starts_zyx: Array of shape (N, 3) with the ZYX start coordinates of each run
        lengths: Array of shape (N,) with the length of each run
        num_points: Number of points to sample
        
    Returns:
        Array of shape (num_points, 3) with the ZYX coordinates of sampled points
    """
    # Sample rows with probability proportional to run length
    chosen_rows = np.random.choice(
        len(starts_zyx),
        num_points,
        replace=True,
        p=lengths / lengths.sum()
    )
    
    # Get the start coordinates for each sampled row
    points_zyx = starts_zyx[chosen_rows].copy()
    
    # Add random offset in the X dimension (Z in ZYX coordinates)
    points_zyx[:, 2] += np.random.randint(0, lengths[chosen_rows])
    
    return points_zyx