# dvid-point-cloud

Library for creating point clouds for sparse volumes within DVID.

## Installation

```bash
pip install dvid-point-cloud
```

Or install from source:

```bash
git clone https://github.com/username/dvid-point-cloud.git
cd dvid-point-cloud
pip install -e .
```

For development, install with extra dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

Generate a uniform point cloud from a DVID label:

```python
import dvid_point_cloud as dpc

# Define parameters
server = "http://my-dvid-server.janelia.org"
uuid = "bc9a0f"  # Hexadecimal string identifying the version
label_id = 189310  # The neuron/segment ID
density = 0.01  # Sample 1% of the voxels

# Generate the point cloud
point_cloud = dpc.uniform_sample(server, uuid, label_id, density)

# point_cloud is a numpy array with shape (N, 3) 
# where each row is an XYZ coordinate
print(f"Generated a point cloud with {len(point_cloud)} points")
```

## Requirements

- Python 3.7+
- numpy
- requests
- protobuf

## Building Protocol Buffers

The protocol buffer definitions need to be compiled before use:

```bash
cd dvid_point_cloud/proto
protoc --python_out=. labelindex.proto
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=dvid_point_cloud

# Run a specific test file
pytest tests/test_sampling.py

# Run a specific test
pytest tests/test_sampling.py::test_uniform_sample_integration
```

### Linting and Type Checking

```bash
# Run linter
flake8 dvid_point_cloud

# Run type checker
mypy dvid_point_cloud
```
