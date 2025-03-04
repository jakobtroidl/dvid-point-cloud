"""Pytest configuration for dvid-point-cloud tests."""

import os
import struct
import tempfile
from typing import Dict, List, Tuple, Union

import numpy as np
import pytest
import requests_mock


@pytest.fixture
def mock_server():
    """Create a mock server for DVID API requests."""
    with requests_mock.Mocker() as m:
        yield m




@pytest.fixture
def create_sparse_volume():
    """Create a sparse volume in DVID RLE format for testing."""
    def _create_sparse_volume(runs: List[Tuple[int, int, int, int]]):
        """
        Create a sparse volume in DVID RLE format.
        
        Args:
            runs: List of (x, y, z, length) tuples representing runs of voxels
            
        Returns:
            Binary data in DVID RLE format
        """
        # Create header
        header = bytearray()
        header.append(0)  # Payload descriptor (0 = binary sparse volume)
        header.append(3)  # Number of dimensions
        header.append(0)  # Dimension of run (0 = X)
        header.append(0)  # Reserved byte
        
        # Add voxel count (placeholder)
        header.extend(struct.pack("<I", 0))
        
        # Add number of spans
        header.extend(struct.pack("<I", len(runs)))
        
        # Create body with runs
        body = bytearray()
        for x, y, z, length in runs:
            body.extend(struct.pack("<i", x))
            body.extend(struct.pack("<i", y))
            body.extend(struct.pack("<i", z))
            body.extend(struct.pack("<i", length))
        
        return bytes(header + body)
    
    return _create_sparse_volume


@pytest.fixture
def generate_sparse_volume():
    """Generate a random sparse volume for fuzz testing."""
    def _generate_sparse_volume(size: Tuple[int, int, int], label_id: int, 
                               density: float = 0.2, seed: int = None):
        """
        Generate a random sparse volume with a continuous subset of voxels for a given label.
        
        Args:
            size: Size of the volume (x, y, z)
            label_id: Label ID
            density: Density of the label in the volume (0.0 to 1.0)
            seed: Random seed
            
        Returns:
            Tuple containing:
                - None (placeholder to maintain compatibility)
                - List of runs (x, y, z, length)
                - Total number of voxels in the label
        """
        if seed is not None:
            np.random.seed(seed)
            
        x_size, y_size, z_size = size
        
        # Generate a binary mask for the label (1 = label present, 0 = background)
        volume = np.zeros(size, dtype=np.bool_)
        
        # Create a continuous region by starting at a random point and growing outward
        center = (np.random.randint(0, x_size),
                  np.random.randint(0, y_size),
                  np.random.randint(0, z_size))
        
        # Determine how many voxels should be in the label
        total_voxels = x_size * y_size * z_size
        target_voxels = int(total_voxels * density)
        
        # Start with the center voxel
        volume[center[0], center[1], center[2]] = True
        current_voxels = 1
        
        # Queue of neighboring voxels to potentially add
        neighbors = []
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nx, ny, nz = center[0] + dx, center[1] + dy, center[2] + dz
            if 0 <= nx < x_size and 0 <= ny < y_size and 0 <= nz < z_size:
                neighbors.append((nx, ny, nz))
        
        # Grow the region until we reach the target size
        while current_voxels < target_voxels and neighbors:
            # Randomly select a neighboring voxel
            idx = np.random.randint(0, len(neighbors))
            x, y, z = neighbors.pop(idx)
            
            # Add it to the label region
            if not volume[x, y, z]:
                volume[x, y, z] = True
                current_voxels += 1
                
                # Add its neighbors to the queue
                for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < x_size and 0 <= ny < y_size and 0 <= nz < z_size and not volume[nx, ny, nz]:
                        neighbors.append((nx, ny, nz))
        
        # Convert volume to runs along X dimension
        runs = []
        for z in range(z_size):
            for y in range(y_size):
                run_start = None
                run_length = 0
                
                for x in range(x_size + 1):  # +1 to handle runs that end at the volume boundary
                    is_label = x < x_size and volume[x, y, z]
                    
                    if is_label:
                        if run_start is None:
                            run_start = x
                        run_length += 1
                    elif run_start is not None:
                        runs.append((run_start, y, z, run_length))
                        run_start = None
                        run_length = 0
        
        return None, runs, current_voxels
    
    return _generate_sparse_volume