% rtcExtractBVH(3) | Embree Ray Tracing Kernels 3

#### NAME

    rtcExtractBVH - Extract BVH from a scene

#### SYNOPSIS

    #include <embree3/rtcore_builder.h>

    struct BVHPrimitive
    {
      unsigned int geomID;
      unsigned int primID;
    };
  
    struct RTCBVHExtractFunction
    {
      void (*expectedSize) (unsigned int num_leaf, unsigned int num_tri, void *userData);
  
      void* (*createLeaf) (unsigned int nbPrim, const BVHPrimitive prims[], void *userData);
      void* (*createInstance) (unsigned int nbPrim, const unsigned int geomID[], void *userData);
      void* (*createCurve) (unsigned int nbPrim, const BVHPrimitive prims[], void *userData);
  
      void* (*createInnerNode) (unsigned int nbChild, void* children[], void *userData);
  
      void (*setAlignedBounds) (void *node, const RTCBounds &bounds, void *userData);
      void (*setLinearBounds) (void *node, const RTCLinearBounds &lbounds, void *userData);
      void (*setUnalignedBounds) (void *node, const RTCAffineSpace &affSpace, void *userData);
      void (*setUnalignedLinearBounds) (void *node, const RTCAffineSpace &affSpace, const RTCBounds &bounds, void *userData);
    };

    void* rtcExtractBVH(
      RTCScene hscene,
      RTCBVHExtractFunction args,
      void *userData
    );

#### DESCRIPTION

The `rtcExtractBVH` function can be used to extract the BVH tree from a scene
that has already been built.

The `rtcExtractBVH` take the scene to export as parameter as well as a
structure that contain callback function pointers and a user-defined pointer
that is passed to all callback functions when invoked.
All callback functions are typically called from a multiple threads, thus
their implementation must be thread-safe.

At least 8 callback functions must be registered, which are invoked during
build to create BVH leaf (`createLeaf`, `createInstance` and `createCurve`
member), to create a BVH inner node (`createInnerNode` member) and to set the
bounding boxes of a node (`setAlignedBounds`, `setLinearBounds`,
`setUnalignedBounds` and `setUnalignedLinearBounds` member).

The function pointer used to preallocate buffer (`expectedSize` member) is
optional and may be `NULL`.

The `createLeaf`, `createInstance` and `createCurve` callback also gets the
number of primitives for this leaf, and an array of size `nbPrim` of
primitive objects.

The `createInnerNode` callback gets the number of child, as well as an array
of `nbChild` child previously returned by either `createLeaf`,
`createInstance` or `createCurve`.

The `setAlignedBounds`, `setLinearBounds`, `setUnalignedBounds` and
`setUnalignedLinearBounds` callback function gets a pointer to the node as
input (`node` argument) as well as the bounds for this node.

The `setAlignedBounds` callback function take bounds of type `RTCBounds` which
represent an aligned bound box.

The `setLinearBounds` callback function take bounds of type `RTCLinearBounds`
which represent an aligned bound box that can be linearly interpolated over
time. The `align0` and `align1` member are used to time range of the node

The `setUnalignedBounds` callback function take bounds of type `RTCAffineSpace`
which represent an affine space in which the bound are ((0, 0, 0), (1, 1, 1)).

The `setUnalignedLinearBounds` callback function take bounds of type
`RTCAffineSpace` and `RTCBounds`. The bound box in the affine space described
by `affSpace` are the linear interpolation between ((0, 0, 0), (1, 1, 1)) and
`bounds`.

#### EXIT STATUS

Return the root node, returned by the root `createInnerNode`, or `NULL` if an
error occurred.
On failure an error code is set that can be queried using `rtcDeviceGetError`.

#### SEE ALSO

[rtcCommitScene]
