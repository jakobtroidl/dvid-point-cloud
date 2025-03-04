"""Functions for sampling point clouds from DVID."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .client import DVIDClient
from .parse import (
    parse_rles, 
    vectorized_sample_from_rles
)

logger = logging.getLogger(__name__)

def uniform_sample(server: str, uuid: str, label_id: int, 
                  density_or_count: Union[float, int],
                  instance: str = "segmentation",
                  scale: int = 0,
                  supervoxels: bool = False,
                  output_format: str = "xyz") -> Union[np.ndarray, pd.DataFrame]:
    """
    Generate a uniform point cloud sample from a DVID label using a vectorized approach.
    
    Args:
        server: DVID server URL
        uuid: UUID of the DVID node
        label_id: Label ID to query
        density_or_count: If 0.0001 <= value <= 1.0, treated as density (fraction of voxels).
                         If value > 1.0, treated as the number of points to sample.
        instance: Name of the labelmap instance (default: "segmentation")
        scale: Scale level at which to fetch the sparsevol (0 is highest resolution)
        supervoxels: If True, fetch supervoxel data instead of body data
        output_format: Output format: "xyz" for numpy array, "dataframe" for pandas DataFrame
        
    Returns:
        If output_format="xyz":
            Numpy array of shape (N, 3) with uniformly sampled XYZ coordinates
        If output_format="dataframe":
            pandas DataFrame with 'x', 'y', 'z' columns
    """
    if isinstance(density_or_count, float) and not 0.0001 <= density_or_count <= 1.0:
        raise ValueError(f"Density must be between 0.0001 and 1.0, got {density_or_count}")
        
    if output_format not in ["xyz", "dataframe"]:
        raise ValueError(f"output_format must be 'xyz' or 'dataframe', got {output_format}")
    
    client = DVIDClient(server)
    
    # Get sparse volume data for the label at the requested scale
    sparse_vol_data = client.get_sparse_vol(
        uuid, instance, label_id, format="rles", scale=scale, supervoxels=supervoxels
    )
    
    # Parse RLEs into starts_zyx and lengths arrays
    starts_zyx, lengths = parse_rles(sparse_vol_data)
    
    # Calculate total # voxels for this label from the RLEs
    total_voxels = lengths.sum()
    logger.info(f"Label {label_id} has {total_voxels} total voxels")
    
    # Determine how many sample points we need
    if 0.0001 <= density_or_count <= 1.0:
        # Treat as density
        density = density_or_count
        num_samples = max(1, int(round(total_voxels * density)))
        logger.info(f"Sampling {num_samples} points at density {density}")
    else:
        # Treat as count
        num_samples = int(density_or_count)
        density = num_samples / total_voxels
        logger.info(f"Sampling {num_samples} points (density: {density:.6f})")
    
    # Generate point cloud using vectorized approach
    points_zyx = vectorized_sample_from_rles(starts_zyx, lengths, num_samples)
    
    # Apply scale factor to convert from downsampled coordinates to full resolution if needed
    if scale > 0:
        points_zyx *= (2**scale)
    
    # Convert from ZYX to XYZ
    points_xyz = points_zyx[:, ::-1]
    
    # Return requested format
    if output_format == "dataframe":
        return pd.DataFrame(points_xyz, columns=["x", "y", "z"])
    else:
        return points_xyz


def sample_for_bodies(server: str, uuid: str, instance: str, body_ids: List[int], 
                     num_points_per_body: int = 1000, scale: int = 0,
                     supervoxels: bool = False) -> Dict[int, np.ndarray]:
    """
    Generate point cloud samples for multiple bodies efficiently.
    
    Args:
        server: DVID server URL
        uuid: UUID of the DVID node
        instance: Name of the labelmap instance
        body_ids: List of body IDs to sample
        num_points_per_body: Number of points to sample for each body
        scale: Scale level at which to fetch the sparsevol (0 is highest resolution)
        supervoxels: If True, fetch supervoxel data instead of body data
        
    Returns:
        Dictionary mapping body IDs to point cloud arrays (each is N×3 with XYZ coordinates)
    """
    client = DVIDClient(server)
    result = {}
    
    for body_id in body_ids:
        try:
            sparse_vol_data = client.get_sparse_vol(
                uuid, instance, body_id, format="rles", scale=scale, supervoxels=supervoxels
            )
            
            starts_zyx, lengths = parse_rles(sparse_vol_data)
            
            # Skip empty bodies or bodies with no RLEs
            if len(lengths) == 0 or lengths.sum() == 0:
                logger.warning(f"Body {body_id} has no voxels, skipping")
                continue
                
            points_zyx = vectorized_sample_from_rles(starts_zyx, lengths, num_points_per_body)
            
            # Apply scale factor
            if scale > 0:
                points_zyx *= (2**scale)
                
            # Convert to XYZ
            result[body_id] = points_zyx[:, ::-1]
            
        except Exception as e:
            logger.error(f"Error sampling points for body {body_id}: {e}")
    
    return result