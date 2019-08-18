// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#ifdef _WIN32
#  define RTC_API extern "C" __declspec(dllexport)
#else
#  define RTC_API extern "C" __attribute__ ((visibility ("default")))
#endif

#include "default.h"
#include "device.h"
#include "scene.h"
#include "context.h"
#include "alloc.h"

#include "../builders/bvh_builder_sah.h"
#include "../builders/bvh_builder_morton.h"

#include "../bvh/bvh.h"
#include "../geometry/instance.h"
#include "../geometry/trianglev.h"
#include "../geometry/trianglei.h"
#include "../geometry/curveNi.h"
#include "../geometry/curveNi_mb.h"
#include "../geometry/linei.h"

namespace embree
{ 
  DECLARE_ISA_FUNCTION(unsigned int, getLine8iPrimId, Line8i* COMMA unsigned int);

  namespace isa // FIXME: support more ISAs for builders
  {
    struct BVH : public RefCount
    {
      BVH (Device* device)
        : device(device), allocator(device,true), morton_src(device,0), morton_tmp(device,0)
      {
        device->refInc();
      }

      ~BVH() {
        device->refDec();
      }

    public:
      Device* device;
      FastAllocator allocator;
      mvector<BVHBuilderMorton::BuildPrim> morton_src;
      mvector<BVHBuilderMorton::BuildPrim> morton_tmp;
    };

    RTC_API RTCBVH rtcNewBVH(RTCDevice device)
    {
      RTC_CATCH_BEGIN;
      RTC_TRACE(rtcNewAllocator);
      RTC_VERIFY_HANDLE(device);
      BVH* bvh = new BVH((Device*)device);
      return (RTCBVH) bvh->refInc();
      RTC_CATCH_END((Device*)device);
      return nullptr;
    }

    void* rtcBuildBVHMorton(const RTCBuildArguments* arguments)
    {
      BVH* bvh = (BVH*) arguments->bvh;
      RTCBuildPrimitive* prims_i =  arguments->primitives;
      size_t primitiveCount = arguments->primitiveCount;
      RTCCreateNodeFunction createNode = arguments->createNode;
      RTCSetNodeChildrenFunction setNodeChildren = arguments->setNodeChildren;
      RTCSetNodeBoundsFunction setNodeBounds = arguments->setNodeBounds;
      RTCCreateLeafFunction createLeaf = arguments->createLeaf;
      RTCProgressMonitorFunction buildProgress = arguments->buildProgress;
      void* userPtr = arguments->userPtr;
        
      std::atomic<size_t> progress(0);
      
      /* initialize temporary arrays for morton builder */
      PrimRef* prims = (PrimRef*) prims_i;
      mvector<BVHBuilderMorton::BuildPrim>& morton_src = bvh->morton_src;
      mvector<BVHBuilderMorton::BuildPrim>& morton_tmp = bvh->morton_tmp;
      morton_src.resize(primitiveCount);
      morton_tmp.resize(primitiveCount);

      /* compute centroid bounds */
      const BBox3fa centBounds = parallel_reduce ( size_t(0), primitiveCount, BBox3fa(empty), [&](const range<size_t>& r) -> BBox3fa {

          BBox3fa bounds(empty);
          for (size_t i=r.begin(); i<r.end(); i++) 
            bounds.extend(prims[i].bounds().center2());
          return bounds;
        }, BBox3fa::merge);
      
      /* compute morton codes */
      BVHBuilderMorton::MortonCodeMapping mapping(centBounds);
      parallel_for ( size_t(0), primitiveCount, [&](const range<size_t>& r) {
          BVHBuilderMorton::MortonCodeGenerator generator(mapping,&morton_src[r.begin()]);
          for (size_t i=r.begin(); i<r.end(); i++) {
            generator(prims[i].bounds(),(unsigned) i);
          }
        });

      /* start morton build */
      std::pair<void*,BBox3fa> root = BVHBuilderMorton::build<std::pair<void*,BBox3fa>>(
        
        /* thread local allocator for fast allocations */
        [&] () -> FastAllocator::CachedAllocator { 
          return bvh->allocator.getCachedAllocator();
        },
        
        /* lambda function that allocates BVH nodes */
        [&] ( const FastAllocator::CachedAllocator& alloc, size_t N ) -> void* {
          return createNode((RTCThreadLocalAllocator)&alloc, (unsigned int)N,userPtr);
        },
        
        /* lambda function that sets bounds */
        [&] (void* node, const std::pair<void*,BBox3fa>* children, size_t N) -> std::pair<void*,BBox3fa>
        {
          BBox3fa bounds = empty;
          void* childptrs[BVHBuilderMorton::MAX_BRANCHING_FACTOR];
          const RTCBounds* cbounds[BVHBuilderMorton::MAX_BRANCHING_FACTOR];
          for (size_t i=0; i<N; i++) {
            bounds.extend(children[i].second);
            childptrs[i] = children[i].first;
            cbounds[i] = (const RTCBounds*)&children[i].second;
          }
          setNodeBounds(node,cbounds,(unsigned int)N,userPtr);
          setNodeChildren(node,childptrs, (unsigned int)N,userPtr);
          return std::make_pair(node,bounds);
        },
        
        /* lambda function that creates BVH leaves */
        [&]( const range<unsigned>& current, const FastAllocator::CachedAllocator& alloc) -> std::pair<void*,BBox3fa>
        {
          const size_t id = morton_src[current.begin()].index;
          const BBox3fa bounds = prims[id].bounds(); 
          void* node = createLeaf((RTCThreadLocalAllocator)&alloc,prims_i+current.begin(),current.size(),userPtr);
          return std::make_pair(node,bounds);
        },
        
        /* lambda that calculates the bounds for some primitive */
        [&] (const BVHBuilderMorton::BuildPrim& morton) -> BBox3fa {
          return prims[morton.index].bounds();
        },
        
        /* progress monitor function */
        [&] (size_t dn) {
          if (!buildProgress) return true;
          const size_t n = progress.fetch_add(dn)+dn;
          const double f = std::min(1.0,double(n)/double(primitiveCount));
          return buildProgress(userPtr,f);
        },
        
        morton_src.data(),morton_tmp.data(),primitiveCount,
        *arguments);

      bvh->allocator.cleanup();
      return root.first;
    }

    void* rtcBuildBVHBinnedSAH(const RTCBuildArguments* arguments)
    {
      BVH* bvh = (BVH*) arguments->bvh;
      RTCBuildPrimitive* prims =  arguments->primitives;
      size_t primitiveCount = arguments->primitiveCount;
      RTCCreateNodeFunction createNode = arguments->createNode;
      RTCSetNodeChildrenFunction setNodeChildren = arguments->setNodeChildren;
      RTCSetNodeBoundsFunction setNodeBounds = arguments->setNodeBounds;
      RTCCreateLeafFunction createLeaf = arguments->createLeaf;
      RTCProgressMonitorFunction buildProgress = arguments->buildProgress;
      void* userPtr = arguments->userPtr;
      
      std::atomic<size_t> progress(0);
  
      /* calculate priminfo */
      auto computeBounds = [&](const range<size_t>& r) -> CentGeomBBox3fa
        {
          CentGeomBBox3fa bounds(empty);
          for (size_t j=r.begin(); j<r.end(); j++)
            bounds.extend((BBox3fa&)prims[j]);
          return bounds;
        };
      const CentGeomBBox3fa bounds = 
        parallel_reduce(size_t(0),primitiveCount,size_t(1024),size_t(1024),CentGeomBBox3fa(empty), computeBounds, CentGeomBBox3fa::merge2);

      const PrimInfo pinfo(0,primitiveCount,bounds);
      
      /* build BVH */
      void* root = BVHBuilderBinnedSAH::build<void*>(
        
        /* thread local allocator for fast allocations */
        [&] () -> FastAllocator::CachedAllocator { 
          return bvh->allocator.getCachedAllocator();
        },

        /* lambda function that creates BVH nodes */
        [&](BVHBuilderBinnedSAH::BuildRecord* children, const size_t N, const FastAllocator::CachedAllocator& alloc) -> void*
        {
          void* node = createNode((RTCThreadLocalAllocator)&alloc, (unsigned int)N,userPtr);
          const RTCBounds* cbounds[GeneralBVHBuilder::MAX_BRANCHING_FACTOR];
          for (size_t i=0; i<N; i++) cbounds[i] = (const RTCBounds*) &children[i].prims.geomBounds;
          setNodeBounds(node,cbounds, (unsigned int)N,userPtr);
          return node;
        },

        /* lambda function that updates BVH nodes */
        [&](const BVHBuilderBinnedSAH::BuildRecord& precord, const BVHBuilderBinnedSAH::BuildRecord* crecords, void* node, void** children, const size_t N) -> void* {
          setNodeChildren(node,children, (unsigned int)N,userPtr);
          return node;
        },
        
        /* lambda function that creates BVH leaves */
        [&](const PrimRef* prims, const range<size_t>& range, const FastAllocator::CachedAllocator& alloc) -> void* {
          return createLeaf((RTCThreadLocalAllocator)&alloc,(RTCBuildPrimitive*)(prims+range.begin()),range.size(),userPtr);
        },
        
        /* progress monitor function */
        [&] (size_t dn) {
          if (!buildProgress) return true;
          const size_t n = progress.fetch_add(dn)+dn;
          const double f = std::min(1.0,double(n)/double(primitiveCount));
          return buildProgress(userPtr,f);
        },
        
        (PrimRef*)prims,pinfo,*arguments);
        
      bvh->allocator.cleanup();
      return root;
    }

    void* rtcBuildBVHSpatialSAH(const RTCBuildArguments* arguments)
    {
      BVH* bvh = (BVH*) arguments->bvh;
      RTCBuildPrimitive* prims =  arguments->primitives;
      size_t primitiveCount = arguments->primitiveCount;
      RTCCreateNodeFunction createNode = arguments->createNode;
      RTCSetNodeChildrenFunction setNodeChildren = arguments->setNodeChildren;
      RTCSetNodeBoundsFunction setNodeBounds = arguments->setNodeBounds;
      RTCCreateLeafFunction createLeaf = arguments->createLeaf;
      RTCSplitPrimitiveFunction splitPrimitive = arguments->splitPrimitive;
      RTCProgressMonitorFunction buildProgress = arguments->buildProgress;
      void* userPtr = arguments->userPtr;
      
      std::atomic<size_t> progress(0);
  
      /* calculate priminfo */
      auto computeBounds = [&](const range<size_t>& r) -> CentGeomBBox3fa
        {
          CentGeomBBox3fa bounds(empty);
          for (size_t j=r.begin(); j<r.end(); j++)
            bounds.extend((BBox3fa&)prims[j]);
          return bounds;
        };
      const CentGeomBBox3fa bounds = 
        parallel_reduce(size_t(0),primitiveCount,size_t(1024),size_t(1024),CentGeomBBox3fa(empty), computeBounds, CentGeomBBox3fa::merge2);

      const PrimInfo pinfo(0,primitiveCount,bounds);

      /* function that splits a build primitive */
      struct Splitter
      {
        Splitter (RTCSplitPrimitiveFunction splitPrimitive, unsigned geomID, unsigned primID, void* userPtr)
          : splitPrimitive(splitPrimitive), geomID(geomID), primID(primID), userPtr(userPtr) {}
        
        __forceinline void operator() (PrimRef& prim, const size_t dim, const float pos, PrimRef& left_o, PrimRef& right_o) const 
        {
          prim.geomIDref() &= BVHBuilderBinnedFastSpatialSAH::GEOMID_MASK;
          splitPrimitive((RTCBuildPrimitive*)&prim,(unsigned)dim,pos,(RTCBounds*)&left_o,(RTCBounds*)&right_o,userPtr);
          left_o.geomIDref()  = geomID; left_o.primIDref()  = primID;
          right_o.geomIDref() = geomID; right_o.primIDref() = primID;
        }

        __forceinline void operator() (const BBox3fa& box, const size_t dim, const float pos, BBox3fa& left_o, BBox3fa& right_o) const 
        {
          PrimRef prim(box,geomID & BVHBuilderBinnedFastSpatialSAH::GEOMID_MASK,primID);
          splitPrimitive((RTCBuildPrimitive*)&prim,(unsigned)dim,pos,(RTCBounds*)&left_o,(RTCBounds*)&right_o,userPtr);
        }
   
        RTCSplitPrimitiveFunction splitPrimitive;
        unsigned geomID;
        unsigned primID;
        void* userPtr;
      };

      /* build BVH */
      void* root = BVHBuilderBinnedFastSpatialSAH::build<void*>(
        
        /* thread local allocator for fast allocations */
        [&] () -> FastAllocator::CachedAllocator { 
          return bvh->allocator.getCachedAllocator();
        },

        /* lambda function that creates BVH nodes */
        [&] (BVHBuilderBinnedFastSpatialSAH::BuildRecord* children, const size_t N, const FastAllocator::CachedAllocator& alloc) -> void*
        {
          void* node = createNode((RTCThreadLocalAllocator)&alloc, (unsigned int)N,userPtr);
          const RTCBounds* cbounds[GeneralBVHBuilder::MAX_BRANCHING_FACTOR];
          for (size_t i=0; i<N; i++) cbounds[i] = (const RTCBounds*) &children[i].prims.geomBounds;
          setNodeBounds(node,cbounds, (unsigned int)N,userPtr);
          return node;
        },

        /* lambda function that updates BVH nodes */
        [&] (const BVHBuilderBinnedFastSpatialSAH::BuildRecord& precord, const BVHBuilderBinnedFastSpatialSAH::BuildRecord* crecords, void* node, void** children, const size_t N) -> void* {
          setNodeChildren(node,children, (unsigned int)N,userPtr);
          return node;
        },
        
        /* lambda function that creates BVH leaves */
        [&] (const PrimRef* prims, const range<size_t>& range, const FastAllocator::CachedAllocator& alloc) -> void* {
          return createLeaf((RTCThreadLocalAllocator)&alloc,(RTCBuildPrimitive*)(prims+range.begin()),range.size(),userPtr);
        },
        
        /* returns the splitter */
        [&] ( const PrimRef& prim ) -> Splitter {
          return Splitter(splitPrimitive,prim.geomID(),prim.primID(),userPtr);
        },

        /* progress monitor function */
        [&] (size_t dn) {
          if (!buildProgress) return true;
          const size_t n = progress.fetch_add(dn)+dn;
          const double f = std::min(1.0,double(n)/double(primitiveCount));
          return buildProgress(userPtr,f);
        },
        
        (PrimRef*)prims,
        arguments->primitiveArrayCapacity,
        pinfo,*arguments);
        
      bvh->allocator.cleanup();
      return root;
    }

    template<int N>
    void* createLeaf(const typename BVHN<N>::NodeRef node,
                     const PrimitiveType *leafType,
                     const RTCBVHExtractFunction args,
                     void *userData) {
        size_t nb;
        if(leafType == &Triangle4v::type) {
            Triangle4v *prims = reinterpret_cast<Triangle4v *>(node.leaf(nb));
            BVHPrimitive *primsArray = (BVHPrimitive *)alloca(4 * nb * sizeof(BVHPrimitive));
            unsigned int realNum = 0;
            for(int i = 0; i < nb; ++i) {
                for(size_t j = 0; j < prims[i].size(); j++) {
                    primsArray[realNum].geomID = prims[i].geomID(j);
                    primsArray[realNum].primID = prims[i].primID(j);
                    ++realNum;
                }
            }

            return args.createLeaf(realNum, primsArray, userData);
        } else if(leafType == &Triangle4i::type) {
            Triangle4i *prims = reinterpret_cast<Triangle4i *>(node.leaf(nb));
            BVHPrimitive *primsArray = (BVHPrimitive *)alloca(4 * nb * sizeof(BVHPrimitive));
            unsigned int realNum = 0;
            for(int i = 0; i < nb; ++i) {
                for(size_t j = 0; j < prims[i].size(); j++) {
                    primsArray[realNum].geomID = prims[i].geomID(j);
                    primsArray[realNum].primID = prims[i].primID(j);
                    ++realNum;
                }
            }

            return args.createLeaf(realNum, primsArray, userData);
        } else if(leafType == &InstancePrimitive::type) {
            InstancePrimitive *prims = reinterpret_cast<InstancePrimitive *>(node.leaf(nb));
            unsigned int *geomIDs = (unsigned int *)alloca(sizeof(unsigned int)*nb);
            for(int i = 0; i < nb; ++i)
                geomIDs[i] = prims[i].instance->geomID;

            return args.createInstance(nb, geomIDs, userData);
        } else if(leafType == &Curve8i::type) {
            typedef unsigned char Primitive;

            Primitive* prim = (Primitive*)node.leaf(nb);
            if(nb == 0) return nullptr;

            assert(nb == 1);
            Geometry::GType ty = (Geometry::GType)(*prim);

            BVHPrimitive primsArray[8];
            unsigned int realNum = 0;

            switch(ty) {
            case Geometry::GTY_FLAT_LINEAR_CURVE: {
                // Access to PrimID from right ISA, otherwise lead to allignement issue
                DEFINE_ISA_FUNCTION(unsigned int, getLine8iPrimId, Line8i* COMMA unsigned int)
                SELECT_SYMBOL_INIT_AVX(getCPUFeatures(), getLine8iPrimId)

                Line8i *line = reinterpret_cast<Line8i*>(prim);

                for(size_t i = 0; i < line->m; i++) {
                    primsArray[realNum].geomID = line->geomID();
                    primsArray[realNum].primID = getLine8iPrimId(line, i);
                    ++realNum;
                }
            } break;
            case Geometry::GTY_FLAT_BEZIER_CURVE:
            case Geometry::GTY_ROUND_BEZIER_CURVE:
            case Geometry::GTY_ORIENTED_BEZIER_CURVE:
            case Geometry::GTY_FLAT_BSPLINE_CURVE:
            case Geometry::GTY_ROUND_BSPLINE_CURVE:
            case Geometry::GTY_ORIENTED_BSPLINE_CURVE:
            case Geometry::GTY_FLAT_HERMITE_CURVE:
            case Geometry::GTY_ROUND_HERMITE_CURVE:
            case Geometry::GTY_ORIENTED_HERMITE_CURVE: {
                Curve8i *curve = reinterpret_cast<Curve8i*>(prim);
                const auto Nb = curve->N;
                for(size_t i = 0; i < Nb; i++) {
                    primsArray[realNum].geomID = curve->geomID(Nb);
                    primsArray[realNum].primID = curve->primID(Nb)[i];
                    ++realNum;
                }
            } break;
            default:
                throw_RTCError(RTC_ERROR_INVALID_OPERATION, "Unexpected curve geom type");
            }

            return args.createCurve(realNum, primsArray, userData);
        } else if(leafType == &Curve8iMB::type) {
            typedef unsigned char Primitive;

            Primitive* prim = (Primitive*)node.leaf(nb);
            if(nb == 0) return nullptr;

            assert(nb == 1);
            Geometry::GType ty = (Geometry::GType)(*prim);

            BVHPrimitive primsArray[8];
            unsigned int realNum = 0;

            switch(ty) {
            case Geometry::GTY_FLAT_LINEAR_CURVE: {
                // Access to PrimID from right ISA, otherwise lead to allignement issue
                DEFINE_ISA_FUNCTION(unsigned int, getLine8iPrimId, Line8i* COMMA unsigned int)
                SELECT_SYMBOL_INIT_AVX(getCPUFeatures(), getLine8iPrimId)

                Line8i *line = reinterpret_cast<Line8i*>(prim);

                for(size_t i = 0; i < line->m; i++) {
                    primsArray[realNum].geomID = line->geomID();
                    primsArray[realNum].primID = getLine8iPrimId(line, i);
                    ++realNum;
                }
            } break;
            case Geometry::GTY_FLAT_BEZIER_CURVE:
            case Geometry::GTY_ROUND_BEZIER_CURVE:
            case Geometry::GTY_ORIENTED_BEZIER_CURVE:
            case Geometry::GTY_FLAT_BSPLINE_CURVE:
            case Geometry::GTY_ROUND_BSPLINE_CURVE:
            case Geometry::GTY_ORIENTED_BSPLINE_CURVE:
            case Geometry::GTY_FLAT_HERMITE_CURVE:
            case Geometry::GTY_ROUND_HERMITE_CURVE:
            case Geometry::GTY_ORIENTED_HERMITE_CURVE: {
                Curve8iMB *curve = reinterpret_cast<Curve8iMB*>(prim);
                const auto Nb = curve->N;
                for(size_t i = 0; i < Nb; i++) {
                    primsArray[realNum].geomID = curve->geomID(Nb);
                    primsArray[realNum].primID = curve->primID(Nb)[i];
                    ++realNum;
                }
            } break;
            default:
                throw_RTCError(RTC_ERROR_INVALID_OPERATION, "Unexpected curve geom type");
            }

            return args.createCurve(realNum, primsArray, userData);
        } else {
            throw_RTCError(RTC_ERROR_INVALID_OPERATION, "Unsupported primitive");
        }
    }

    inline RTCBounds boundsToRTC(const BBox3fa &bounds) {
        RTCBounds bb;
        bb.lower_x = bounds.lower.x;
        bb.lower_y = bounds.lower.y;
        bb.lower_z = bounds.lower.z;

        bb.upper_x = bounds.upper.x;
        bb.upper_y = bounds.upper.y;
        bb.upper_z = bounds.upper.z;

        bb.align0 = 0;
        bb.align1 = 1;

        return bb;
    }

    template <unsigned int N>
    inline RTCAffineSpace affineSpaceToRTC(const AffineSpace3vf<N> affSpaces, unsigned int i) {
      RTCAffineSpace affSpace;

      affSpace.affine[0] = affSpaces.p.x[i];
      affSpace.affine[1] = affSpaces.p.y[i];
      affSpace.affine[2] = affSpaces.p.z[i];

      affSpace.linear[0] = affSpaces.l.vx.x[i];
      affSpace.linear[1] = affSpaces.l.vx.y[i];
      affSpace.linear[2] = affSpaces.l.vx.z[i];
      affSpace.linear[3] = affSpaces.l.vy.x[i];
      affSpace.linear[4] = affSpaces.l.vy.y[i];
      affSpace.linear[5] = affSpaces.l.vy.z[i];
      affSpace.linear[6] = affSpaces.l.vz.x[i];
      affSpace.linear[7] = affSpaces.l.vz.y[i];
      affSpace.linear[8] = affSpaces.l.vz.z[i];

      return affSpace;
    }

    template<int N>
    void* recurse(const typename BVHN<N>::NodeRef node,
                  const PrimitiveType *leafType,
                  const RTCBVHExtractFunction args,
                  void *userData) {
      if(node.isLeaf())
          return createLeaf<N>(node, leafType, args, userData);

      const typename BVHN<N>::BaseNode *bnode = nullptr;
      const typename BVHN<N>::AlignedNode *anode = nullptr;
      const typename BVHN<N>::AlignedNodeMB *anodeMB = nullptr;
      const typename BVHN<N>::AlignedNodeMB4D *anodeMB4D = nullptr;

      const typename BVHN<N>::UnalignedNode *unanode = nullptr;
      const typename BVHN<N>::UnalignedNodeMB *unanodeMB = nullptr;

      if(node.isAlignedNode()) {
          anode = node.alignedNode();
          bnode = anode;
      } else if(node.isAlignedNodeMB()) {
          anodeMB = node.alignedNodeMB();
          bnode = anodeMB;
      } else if (node.isAlignedNodeMB4D()) {
          anodeMB4D = node.alignedNodeMB4D();
          anodeMB = anodeMB4D;
          bnode = anodeMB;
      } else if (node.isUnalignedNode()) {
          unanode = node.unalignedNode();
          bnode = unanode;
      } else if (node.isUnalignedNodeMB()) {
          unanodeMB = node.unalignedNodeMB();
          bnode = unanodeMB;
      } else {
          throw_RTCError(RTC_ERROR_INVALID_OPERATION, "Unknown node type");
      }

      unsigned int nb = 0;
      void *children[4];
      for(unsigned int i = 0; i < 4; i++) {
          void *child = recurse<N>(bnode->child(i), leafType, args, userData);
          if(child == nullptr) continue;

          if(anode != nullptr) {
              args.setAlignedBounds(child, boundsToRTC(anode->bounds(i)), userData);
          } else if (anodeMB != nullptr) {
              RTCLinearBounds lb;
              lb.bounds0 = boundsToRTC(anodeMB->bounds0(i));
              lb.bounds1 = boundsToRTC(anodeMB->bounds1(i) - anodeMB->bounds0(i));

              if (anodeMB4D != nullptr) {
                  lb.bounds0.align0 = anodeMB4D->timeRange(i).lower;
                  lb.bounds0.align1 = anodeMB4D->timeRange(i).upper;
              }

              args.setLinearBounds(child, lb, userData);
          } else if(unanode != nullptr) {
              RTCAffineSpace affSpace = affineSpaceToRTC<N>(unanode->naabb, i);
              args.setUnalignedBounds(child, affSpace, userData);
          } else if(unanodeMB != nullptr) {
              RTCAffineSpace affSpace = affineSpaceToRTC<N>(unanodeMB->space0, i);

              RTCBounds bounds;
              bounds.lower_x = unanodeMB->b1.lower.x[i];
              bounds.lower_y = unanodeMB->b1.lower.y[i];
              bounds.lower_z = unanodeMB->b1.lower.z[i];

              bounds.upper_x = unanodeMB->b1.upper.x[i];
              bounds.upper_y = unanodeMB->b1.upper.y[i];
              bounds.upper_z = unanodeMB->b1.upper.z[i];

              args.setUnalignedLinearBounds(child, affSpace, bounds, userData);
          }

          children[nb++] = child;
      }

      return args.createInnerNode(nb, children, userData);
    }

    std::vector<void*> prerecurse(Accel *a, RTCBVHExtractFunction args, void *userData) {
      std::vector<void*> nodes;

      AccelData *ad = a->intersectors.ptr;
      switch(ad->type) {
      case AccelData::TY_BVH4: {
        BVH4 *bvh = dynamic_cast<BVH4 *>(ad);
        BVH4::NodeRef root = bvh->root;

        void* node = recurse<4>(root, bvh->primTy, args, userData);
        args.setAlignedBounds(node, boundsToRTC(bvh->bounds.bounds()), userData);
        nodes.push_back(node);
      } break;
      case AccelData::TY_BVH8: {
        BVH8 *bvh = dynamic_cast<BVH8 *>(ad);
        BVH8::NodeRef root = bvh->root;

        void *node = recurse<8>(root, bvh->primTy, args, userData);
        args.setAlignedBounds(node, boundsToRTC(bvh->bounds.bounds()), userData);
        nodes.push_back(node);
      } break;
      case AccelData::TY_ACCELN: {
        AccelN *acceln = dynamic_cast<AccelN *>(ad);
        for (Accel *acc : acceln->accels) {
          auto newNodes = prerecurse(acc, args, userData);
          nodes.insert(nodes.end(), newNodes.begin(), newNodes.end());
        }
      } break;
      default:
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "Unable to extract something else than BVH4/8 tree");
      }

      return nodes;
    }

    RTC_API void *rtcExtractBVH(RTCScene hscene, RTCBVHExtractFunction args, void *userData) {
      Scene* scene = (Scene*) hscene;
      RTC_CATCH_BEGIN;
      RTC_TRACE(rtcSampleTry);
#if defined(DEBUG)
      RTC_VERIFY_HANDLE(hscene);
#endif

      if (args.expectedSize != nullptr) {
        /* Defines size and helper macro */
        struct Size {
          unsigned int num_prim = 0;
          unsigned int num_tri = 0;
        };

#define SIZE(ops)                                 \
  Size* size = reinterpret_cast<Size*>(userData); \
  ops                                             \
  return nullptr;

#define INC(var, val) size->var += (val);


        Size size;
        RTCBVHExtractFunction param;

        param.createLeaf = [](unsigned int nbPrim, const BVHPrimitive[], void* userData) -> void* {
          SIZE(INC(num_prim, nbPrim) INC(num_tri, 3 * nbPrim))
        };
        param.createInstance = [](unsigned int nbPrim, const unsigned int[], void* userData) -> void* {
          SIZE(INC(num_prim, nbPrim))
        };
        param.createCurve = [](unsigned int nbPrim, const BVHPrimitive[], void* userData) -> void* {
          SIZE(INC(num_prim, nbPrim))
        };
        param.createInnerNode = [](unsigned int, void*[], void* ) -> void* { return nullptr; };
        param.setAlignedBounds = [](void*, const RTCBounds &, void*) {};
        param.setLinearBounds = [](void*, const RTCLinearBounds &, void*) {};
        param.setUnalignedBounds = [](void*, const RTCAffineSpace &, void*) {};
        param.setUnalignedLinearBounds = [](void*, const RTCAffineSpace &, const RTCBounds &, void*) {};
        param.expectedSize = nullptr;

        prerecurse(scene, param, &size);
        args.expectedSize(size.num_prim, size.num_tri, userData);
      }

      std::vector<void*> nodes;

      nodes = prerecurse(scene, args, userData);

      if(nodes.size() == 0)
        return nullptr;

      if(nodes.size() == 1)
        return nodes[0];

      RTCBounds bounds;
      rtcGetSceneBounds(hscene, &bounds);

      void *root = args.createInnerNode(nodes.size(), nodes.data(), userData);
      args.setAlignedBounds(root, bounds, userData);
      return root;

      RTC_CATCH_END2(scene);
      return nullptr;
    }
  }
}
