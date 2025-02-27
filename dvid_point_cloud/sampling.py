"""Functions for sampling point clouds from DVID."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .client import DVIDClient
from .parse import decode_block_coord, parse_rles, rles_to_points

logger = logging.getLogger(__name__)


def count_total_voxels(label_index_data: bytes) -> int:
    """
    Count the total number of voxels in a label from its index.
    
    Args:
        label_index_data: Protobuf serialized LabelIndex data
        
    Returns:
        Total number of voxels in the label
    """
    # For now, we'll use a placeholder implementation until we parse the protobuf
    # A real implementation would parse the protobuf LabelIndex message 
    # and sum all supervoxel counts for all blocks
    
    # Import here to avoid circular imports
    from .proto.labelindex_pb2 import LabelIndex
    
    label_index = LabelIndex()
    label_index.ParseFromString(label_index_data)
    
    total_voxels = 0
    for block_id, sv_count in label_index.blocks.items():
        for _, count in sv_count.counts.items():
            total_voxels += count
    
    return total_voxels


def uniform_sample(server: str, uuid: str, label_id: int, 
                   density: float, instance: str = "segmentation") -> np.ndarray:
    """
    Generate a uniform point cloud sample from a DVID label.
    
    Args:
        server: DVID server URL
        uuid: UUID of the DVID node
        label_id: Label ID to query
        density: Sampling density (0.0001 to 1.0)
        instance: Name of the labelmap instance (default: "segmentation")
        
    Returns:
        Numpy array of shape (N, 3) with uniformly sampled XYZ coordinates
    """
    if not 0.0001 <= density <= 1.0:
        raise ValueError(f"Density must be between 0.0001 and 1.0, got {density}")
    
    client = DVIDClient(server)
    
    # Step 1: Get label index data which tells us which blocks contain our label
    # and how many voxels from the label are in each block
    label_index_data = client.get_label_index(uuid, instance, label_id)
    
    # Step 2: Calculate total voxels for this label
    total_voxels = count_total_voxels(label_index_data)
    logger.info(f"Label {label_id} has {total_voxels} total voxels")
    
    # Step 3: Determine how many sample points we need based on density
    num_samples = max(1, int(round(total_voxels * density)))
    logger.info(f"Sampling {num_samples} points at density {density}")
    
    # Step 4: Generate random sample indices
    # These are indices into the array of all voxels (if we had one)
    sample_indices = np.sort(np.random.choice(total_voxels, size=num_samples, replace=False))
    
    # Step 5: Get sparse volume data for the label
    sparse_vol_data = client.get_sparse_vol(uuid, instance, label_id, format="rles")
    
    # Step 6: Parse RLEs to get runs of voxels
    runs = parse_rles(sparse_vol_data)
    
    # Step 7: Convert runs to samples based on sample indices
    point_cloud = rles_to_points(runs, total_voxels, sample_indices)
    
    return point_cloud