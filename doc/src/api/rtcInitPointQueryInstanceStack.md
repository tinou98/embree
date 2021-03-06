% rtcInitPointQueryInstanceStack(3) | Embree Ray Tracing Kernels 3

#### NAME

    rtcInitPointQueryInstanceStack - initializes the stack of (multilevel-)instance
      information for point queries

#### SYNOPSIS

    #include <embree3/rtcore.h>

    struct RTC_ALIGN(16) RTCPointQueryInstanceStack
    {
      // accumulated 4x4 column major matrices from world to instance space.
      float world2inst[RTC_MAX_INSTANCE_LEVEL_COUNT][16];
      
      // accumulated 4x4 column major matrices from instance to world space.
      float inst2world[RTC_MAX_INSTANCE_LEVEL_COUNT][16];

      // instance ids.
      unsigned int instID[RTC_MAX_INSTANCE_LEVEL_COUNT];
      
      // number of instances currently on the stack.
      unsigned int size;
    };

    void rtcInitPointQueryInstanceStack(
      struct RTCPointQueryInstanceStack* instStack
    );

#### DESCRIPTION

A stack (`RTCPointQueryInstanceStack` type) which stores the IDs and instance
transformations during a BVH traversal for a point query. 

The `rtcInitPointQueryStack` function initializes the stack to
default values and should be called for initialization.

The stack will be passed as an argument to the point query callback function
(see [rtcSetGeometryPointQueryFunction]) and should be used to pass instance
information down the instancing chain for user defined instancing (see
tutorial [ClosestPoint] for a reference implementation of point queries with
user defined instancing).

The stack is an necessary argument to [rtcPointQuery] and Embree internally
uses the topmost instance tranformation of the stack to transform the point
query into instance space.

#### EXIT STATUS

No error code is set by this function.

#### SEE ALSO

[rtcPointQuery], [rtcSetGeometryPointQueryFunction]
