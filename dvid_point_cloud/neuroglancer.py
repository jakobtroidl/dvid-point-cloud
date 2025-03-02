"""Functions for working with Neuroglancer for point cloud visualization."""

from typing import Dict, List, Optional, Union
import copy
from textwrap import dedent, indent
import uuid

import numpy as np
import pandas as pd

# Default shader for point clouds
DEFAULT_POINT_SHADER = """\
void main() {
    setPointMarkerSize(1.0);
    setPointMarkerColor(vec4(defaultColor(), 1.0));
    setPointMarkerBorderWidth(1.0);
    setPointMarkerBorderColor(defaultColor());
}
"""

# Base template for annotation layer JSON
ANNOTATION_LAYER_TEMPLATE = {
    "name": "point-cloud",
    "type": "annotation",
    "source": {
        "url": "local://annotations",
        "transform": {
            "outputDimensions": {
                "x": [
                    8e-09,
                    "m"
                ],
                "y": [
                    8e-09,
                    "m"
                ],
                "z": [
                    8e-09,
                    "m"
                ]
            }
        }
    },
    "tool": "annotatePoint",
    "annotationColor": "#ffffff",
    "shader": DEFAULT_POINT_SHADER,
    "annotations": []
}


def point_cloud_to_neuroglancer_json(
    points: Union[np.ndarray, pd.DataFrame],
    name: str = "point-cloud",
    color: str = "#ffffff",
    shader: Optional[str] = None,
    point_size: float = 1.0,
    res_nm_xyz: List[float] = [8, 8, 8]
) -> Dict:
    """
    Convert a point cloud to Neuroglancer annotation layer JSON.
    
    Args:
        points: Point cloud data, either as:
            - numpy array of shape (N, 3) with XYZ coordinates
            - pandas DataFrame with 'x', 'y', 'z' columns or 'z', 'y', 'x' columns
        name: Name of the annotation layer
        color: Hex color string (e.g., "#ffffff" for white)
        shader: Custom shader code. If None, uses the default point shader
        point_size: Size of the point markers in Neuroglancer
        res_nm_xyz: Resolution in nanometers [x, y, z]
        
    Returns:
        Dictionary containing the Neuroglancer annotation layer JSON
    """
    # Convert points to a standard format
    if isinstance(points, np.ndarray):
        if points.shape[1] == 3:
            # Assume XYZ order for numpy arrays
            df = pd.DataFrame(points, columns=["x", "y", "z"])
        else:
            raise ValueError(f"Expected points array with shape (N, 3), got {points.shape}")
    elif isinstance(points, pd.DataFrame):
        df = points.copy()
        # Handle both XYZ and ZYX column orders
        if all(col in df.columns for col in ["z", "y", "x"]) and not all(col in df.columns for col in ["x", "y", "z"]):
            # Convert from ZYX to XYZ
            df = df.rename(columns={"z": "x", "y": "y", "x": "z"})
            df = df[["z", "y", "x"]].rename(columns={"z": "x", "y": "y", "x": "z"})
    else:
        raise TypeError(f"Expected numpy array or pandas DataFrame, got {type(points)}")
    
    # Generate a custom shader if needed
    if shader is None:
        shader = _create_point_shader(point_size)
    
    # Create the layer JSON
    layer_json = copy.deepcopy(ANNOTATION_LAYER_TEMPLATE)
    layer_json["name"] = name
    layer_json["annotationColor"] = color
    layer_json["shader"] = shader
    
    # Set the resolution
    res_m = (np.array(res_nm_xyz) * 1e-9).tolist()
    output_dim = {k: [r, 'm'] for k, r in zip('xyz', res_m)}
    layer_json["source"]["transform"]["outputDimensions"] = output_dim
    
    # Create annotation entries
    annotations = []
    for _, row in df.iterrows():
        annotation = {
            "type": "point",
            "point": [int(row["x"]), int(row["y"]), int(row["z"])],
            "id": str(uuid.uuid4())
        }
        annotations.append(annotation)
    
    layer_json["annotations"] = annotations
    return layer_json


def _create_point_shader(point_size: float = 1.0) -> str:
    """
    Create a shader for point visualization in Neuroglancer.
    
    Args:
        point_size: Size of the point markers
        
    Returns:
        Shader code as a string
    """
    shader_body = dedent(f"""\
        //
        // Point Marker API
        //
        setPointMarkerSize({point_size});
        setPointMarkerColor(vec4(defaultColor(), 1.0));
        setPointMarkerBorderWidth(1.0);
        setPointMarkerBorderColor(defaultColor());
    """)
    
    shader_main = dedent(f"""\
        void main() {{
            {indent(shader_body, ' '*12)[12:]}\
        }}
    """)
    
    return shader_main