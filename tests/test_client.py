"""Tests for the client module."""

import json
import numpy as np
import pytest

from dvid_point_cloud.client import DVIDClient


def test_dvid_client_initialization():
    """Test that the DVIDClient is initialized correctly."""
    server = "http://test-server.org"
    client = DVIDClient(server)
    
    assert client.server == server
    assert client.timeout == 60  # Default timeout
    
    # Test with custom timeout
    client = DVIDClient(server, timeout=120)
    assert client.timeout == 120


def test_get_label_index(mock_server):
    """Test that the get_label_index method makes the correct HTTP request."""
    server = "http://test-server.org"
    uuid = "abc123"
    instance = "segmentation"
    label_id = 42
    
    # Create mock response content
    mock_content = b"mock protobuf data"
    
    # Configure mock server
    mock_server.get(f"{server}/api/node/{uuid}/{instance}/index/{label_id}",
                  content=mock_content)
    
    # Call the method
    client = DVIDClient(server)
    response = client.get_label_index(uuid, instance, label_id)
    
    # Check that the response is correct
    assert response == mock_content
    
    # Check that the request was made correctly
    assert mock_server.called
    assert mock_server.call_count == 1
    
    request = mock_server.request_history[0]
    assert request.method == "GET"
    assert request.url == f"{server}/api/node/{uuid}/{instance}/index/{label_id}"


def test_get_sparse_vol(mock_server):
    """Test that the get_sparse_vol method makes the correct HTTP request."""
    server = "http://test-server.org"
    uuid = "abc123"
    instance = "segmentation"
    label_id = 42
    
    # Create mock response content
    mock_content = b"mock sparse volume data"
    
    # Configure mock server
    mock_server.get(f"{server}/api/node/{uuid}/{instance}/sparsevol/{label_id}?format=rles",
                  content=mock_content)
    
    # Call the method
    client = DVIDClient(server)
    response = client.get_sparse_vol(uuid, instance, label_id, format="rles")
    
    # Check that the response is correct
    assert response == mock_content
    
    # Check that the request was made correctly
    assert mock_server.called
    assert mock_server.call_count == 1
    
    request = mock_server.request_history[0]
    assert request.method == "GET"
    assert request.url == f"{server}/api/node/{uuid}/{instance}/sparsevol/{label_id}?format=rles"


def test_get_info(mock_server):
    """Test that the get_info method makes the correct HTTP request."""
    server = "http://test-server.org"
    uuid = "abc123"
    instance = "segmentation"
    
    # Create mock response content
    mock_info = {
        "Base": {
            "TypeName": "labelmap",
            "Name": "segmentation"
        },
        "Extended": {
            "BlockSize": [64, 64, 64],
            "VoxelSize": [8, 8, 8],
            "VoxelUnits": ["nanometers", "nanometers", "nanometers"]
        }
    }
    
    # Configure mock server
    mock_server.get(f"{server}/api/node/{uuid}/{instance}/info",
                  json=mock_info)
    
    # Call the method
    client = DVIDClient(server)
    response = client.get_info(uuid, instance)
    
    # Check that the response is correct
    assert response == mock_info
    
    # Check that the request was made correctly
    assert mock_server.called
    assert mock_server.call_count == 1
    
    request = mock_server.request_history[0]
    assert request.method == "GET"
    assert request.url == f"{server}/api/node/{uuid}/{instance}/info"


def test_get_label_blocks(mock_server):
    """Test that the get_label_blocks method makes the correct HTTP request."""
    server = "http://test-server.org"
    uuid = "abc123"
    instance = "segmentation"
    block_coords = "0,0,0,1,1,1"
    
    # Create mock response content
    mock_content = b"mock block data"
    
    # Configure mock server
    mock_server.get(f"{server}/api/node/{uuid}/{instance}/specificblocks?blocks={block_coords}&scale=0",
                  content=mock_content)
    
    # Call the method
    client = DVIDClient(server)
    response = client.get_label_blocks(uuid, instance, block_coords)
    
    # Check that the response is correct
    assert response == mock_content
    
    # Check that the request was made correctly
    assert mock_server.called
    assert mock_server.call_count == 1
    
    request = mock_server.request_history[0]
    assert request.method == "GET"
    assert request.url == f"{server}/api/node/{uuid}/{instance}/specificblocks?blocks={block_coords}&scale=0"
    
    # Test with custom scale and supervoxels params
    mock_server.get(f"{server}/api/node/{uuid}/{instance}/specificblocks?blocks={block_coords}&scale=2&supervoxels=true",
                  content=mock_content)
    
    response = client.get_label_blocks(uuid, instance, block_coords, scale=2, supervoxels=True)
    
    assert response == mock_content
    assert mock_server.call_count == 2
    
    request = mock_server.request_history[1]
    assert request.method == "GET"
    assert request.url == f"{server}/api/node/{uuid}/{instance}/specificblocks?blocks={block_coords}&scale=2&supervoxels=true"