# CLAUDE_INSTRUCTIONS.md - Development Process with Claude

This document captures the conversation and instructions between Claude and the developer during the creation of this repository.

## Initial Project Setup

**Human:** This will be a python library that I'll have others pip install. It will have to make http calls to a data service for which I have documentation of the HTTP calls. Please setup the normal files I'll need to make it pip installable and also get data via HTTP API, so maybe requests library?

**Claude:** I'll set up the basic structure for a pip-installable Python library that can make HTTP calls using the requests library.

## Understanding the DVID HTTP API

**Human:** I'd like to inform you or provide a github path to where the documentation for the HTTP API lives. How can I do that?

**Claude:** You can share the HTTP API documentation with me in several ways:

1. Share a GitHub URL where the documentation is hosted
2. Copy and paste relevant parts of the documentation
3. Share a local file path if you have the documentation downloaded

Once you provide the documentation, I can tailor the library implementation to match the specific DVID API endpoints and data structures.

**Human:** The HTTP API is described by files within the repo here: https://github.com/janelia-flyem/dvid. In particular, we need to:

**Human:** We'll be interacting via the HTTP APIs for getting metadata and particularly dealing with the `labelmap` datatype. Each datatype has a separate HTTP API defined in the `datatype/<datatype>/<datatype>.go` file as a large text string set to `helpMessage`. We are interested in the `labelmap` datatype so look at the const `helpMessage` set in `datatype/labelmap/labelmap.go`.

Note that all calls will require a UUID that gives the "version" of the data as well as the string for a DVID server, so HTTP requests will be of the form: `http://<server>/api/node/<UUID>/<data instance name>/...`

Usually the `<data instance name>` will be "segmentation" since we are getting label data organized as smaller 3d 64x64x64 chunks that tile a much larger 3d space.

## Implementation Requirements

**Human:** 
1. Given a neuron uint64 (the label or body ID), we will get a `LabelIndex` defined by a protobuf serialization via the `/api/node/<UUID>/segmentation/index/<label ID>`.

2. The `LabeIndex` will provide a map of block coordinates (each block being a 64x64x64 ZYX array of uint64 labels) mapped to a SVCount protobuf message. Each block's SVCount will list provide a mapping of supervoxel IDs to the number of voxels with that ID in the block. Each neuron, which is designated by a label ID, could contain any number of supervoxels (each with a separate supervoxel ID). But all we really care about is for each block, how many voxels really belong to the given label ID, which is the sum of all the counts in SVCount.

3. Now that we have for the given label the distribution of blocks (through the block coordinates) and for each block the number of voxels belonging to the label, we can determine how many sample points will be in each block from the sample density parameter (SD = 0.0001 to 1.0). From the LabelIndex we know the number of voxels, N, for that label, so we can compute how many sample points as SP = N * SD rounded to nearest integer. We'll then randomly select points by having an array of SP indices (integers) that uniquely specify which of our N voxels are selected among the 0..N-1 points.

4. We'll then traverse each block in SVCount by block coordinate, read it from the DVID server using the `/api/node/<UUID>/segmentation/sparsevol/<label ID>` endpoint, maybe starting with "rles" format. As we read block after block, we'll keep track of the number of voxels we've traverse across all blocks as well as iterate through the array of sample point indices. If sample point indices match the index of voxels within the current block, we calculate the point coordinate from the ZYX block coordinate and factor in each block is 64 x 64 x 64 voxels. This gives us a 3d point coordinate for each chosen sample.

**Human:** 

Ideally, the interface to this package would look like:
```python
import dvid_point_cloud as dpc

server = "http://my-dvid-server.janelia.org"
uuid = "bc9a0f"  # The version is a hexadecimal string
label_id = 189310
density = 0.01
point_cloud = dpc.uniform_sample(server, uuid, label_id, density)
```

The `point_cloud` return value should be a numpy array of shape (SP, 3) corresponding to the XYZ point (voxel) coordinates. There will eventually be other sampling methods aside from uniformly sampling across a sparse volume, e.g., sampling from only the surface of a sparse volume.

**Human:** 

For the ZYX block coordinates encoded as uint64, all coordinates are assumed to be positive. The voxel space is defined as starting at (0,0,0) and going to (max_x-1, max_y-1, max_z-1) where max_x, max_y, max_z are the maximum coordinates in X, Y, and Z axis.

## Additional Files and Code Organization

**Claude:** I'll implement the library with the described workflow for creating point clouds from DVID data by:

1. Creating a client module to handle HTTP requests to the DVID server
2. Implementing protobuf definitions based on the DVID documentation
3. Creating parsing functions to handle binary data formats 
4. Implementing the uniform_sample function with the specified interface

**Human:** There is already protobuf definitions provided in the documentation for the GET /index request within the labelmap.go file.

**Claude:** (After finding the protobuf definitions in the DVID codebase)

The LabelIndex protobuf message is defined in the documentation as:

```
message SVCount {
    map<uint64, uint32> counts = 1;
}

message LabelIndex {
    map<uint64, SVCount> blocks = 1;  // key is encoded block coord ZYX (packed little-endian 21-bit numbers where MSB is sign flag)
    uint64 label = 2;
    uint64 last_mut_id = 3;
    string last_mod_time = 4;  // string is time in RFC 3339 format
    string last_mod_user = 5;
    string last_mod_app = 6;
}
```

I'll use these definitions to create the appropriate protobuf files for our library.

## Repository Organization

**Human:** I notice we have a `tmp` directory with some needed code for reference. We should have a .gitignore to avoid that from being committed since this source code is a git repo.

**Claude:** You're absolutely right. I'll create a .gitignore file to exclude the tmp directory and other common files we don't want to commit.

---

These instructions guided the development of the dvid-point-cloud library, which provides functionality for creating point clouds from sparse volumes stored in DVID. The library implements a uniform sampling approach and follows the workflow described above.