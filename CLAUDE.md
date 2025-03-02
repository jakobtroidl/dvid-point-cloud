# CLAUDE.md - Development Guide for dvid-point-cloud

## Build and Test Commands
- Generate protobuf files: `cd dvid_point_cloud/proto && protoc --python_out=. labelindex.proto`
- Install locally for development: `pip install -e .`
- Run tests: `pytest tests/`
- Run single test: `pytest tests/test_file.py::test_function`
- Lint code: `flake8 dvid_point_cloud`
- Type check: `mypy dvid_point_cloud`

## Code Style Guidelines
- Use 4 spaces for indentation
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Module imports should be organized in three groups: standard library, third-party, local
- Use descriptive variable names
- Document code using docstrings in Google style format
- Handle errors with appropriate exception types
- Use logging instead of print statements for diagnostics
- Use numpy arrays for point data to optimize memory usage and performance

## Project Structure
- DVID API endpoints are accessed through the client.py module
- Protocol buffer definitions in proto/
- Core point cloud sampling algorithms in sampling.py
- Parsing and utility functions in parse.py