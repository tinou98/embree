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

#define RTC_EXPORT_API

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
#include "../geometry/linei.h"

namespace embree
{ 
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
	  RTCBuildPrimitive localBuildPrims[RTC_BUILD_MAX_PRIMITIVES_PER_LEAF];
	  BBox3fa bounds = empty;
	  for (size_t i=0;i<current.size();i++)
	    {
	      const size_t id = morton_src[current.begin()+i].index;
	      bounds.extend(prims[id].bounds());
	      localBuildPrims[i] = prims_i[id];
	    }
          void* node = createLeaf((RTCThreadLocalAllocator)&alloc,localBuildPrims,current.size(),userPtr);
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

    static __forceinline const std::pair<CentGeomBBox3fa,unsigned int> mergePair(const std::pair<CentGeomBBox3fa,unsigned int>& a, const std::pair<CentGeomBBox3fa,unsigned int>& b) {
      CentGeomBBox3fa centBounds = CentGeomBBox3fa::merge2(a.first,b.first);
      unsigned int maxGeomID = max(a.second,b.second); 
      return std::pair<CentGeomBBox3fa,unsigned int>(centBounds,maxGeomID);
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

      auto computeBounds = [&](const range<size_t>& r) -> std::pair<CentGeomBBox3fa,unsigned int>
        {
          CentGeomBBox3fa bounds(empty);
          unsigned maxGeomID = 0;
          for (size_t j=r.begin(); j<r.end(); j++)
          {
            bounds.extend((BBox3fa&)prims[j]);
            maxGeomID = max(maxGeomID,prims[j].geomID);
          }
          return std::pair<CentGeomBBox3fa,unsigned int>(bounds,maxGeomID);
        };


      const std::pair<CentGeomBBox3fa,unsigned int> pair = 
        parallel_reduce(size_t(0),primitiveCount,size_t(1024),size_t(1024),std::pair<CentGeomBBox3fa,unsigned int>(CentGeomBBox3fa(empty),0), computeBounds, mergePair);

      CentGeomBBox3fa bounds = pair.first;
      const unsigned int maxGeomID = pair.second;
      
      if (unlikely(maxGeomID >= ((unsigned int)1 << (32-RESERVED_NUM_SPATIAL_SPLITS_GEOMID_BITS))))
        {
          /* fallback code for max geomID larger than threshold */
          return rtcBuildBVHBinnedSAH(arguments);
        }

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
  }

  DECLARE_ISA_FUNCTION(unsigned int, getLine8iPrimId, Line8i* COMMA unsigned int);
}

using namespace embree;
using namespace embree::isa;

RTC_NAMESPACE_BEGIN

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

    RTC_API void* rtcBuildBVH(const RTCBuildArguments* arguments)
    {
      BVH* bvh = (BVH*) arguments->bvh;
      RTC_CATCH_BEGIN;
      RTC_TRACE(rtcBuildBVH);
      RTC_VERIFY_HANDLE(bvh);
      RTC_VERIFY_HANDLE(arguments);
      RTC_VERIFY_HANDLE(arguments->createNode);
      RTC_VERIFY_HANDLE(arguments->setNodeChildren);
      RTC_VERIFY_HANDLE(arguments->setNodeBounds);
      RTC_VERIFY_HANDLE(arguments->createLeaf);

      if (arguments->primitiveArrayCapacity < arguments->primitiveCount)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT,"primitiveArrayCapacity must be greater or equal to primitiveCount")

      /* initialize the allocator */
      bvh->allocator.init_estimate(arguments->primitiveCount*sizeof(BBox3fa));
      bvh->allocator.reset();

      /* switch between differnet builders based on quality level */
      if (arguments->buildQuality == RTC_BUILD_QUALITY_LOW)
        return rtcBuildBVHMorton(arguments);
      else if (arguments->buildQuality == RTC_BUILD_QUALITY_MEDIUM)
        return rtcBuildBVHBinnedSAH(arguments);
      else if (arguments->buildQuality == RTC_BUILD_QUALITY_HIGH) {
        if (arguments->splitPrimitive == nullptr || arguments->primitiveArrayCapacity <= arguments->primitiveCount)
          return rtcBuildBVHBinnedSAH(arguments);
        else
          return rtcBuildBVHSpatialSAH(arguments);
      }
      else
        throw_RTCError(RTC_ERROR_INVALID_OPERATION,"invalid build quality");

      /* if we are in dynamic mode, then do not clear temporary data */
      if (!(arguments->buildFlags & RTC_BUILD_FLAG_DYNAMIC))
      {
        bvh->morton_src.clear();
        bvh->morton_tmp.clear();
      }

      RTC_CATCH_END(bvh->device);
      return nullptr;
    }

    RTC_API void* rtcThreadLocalAlloc(RTCThreadLocalAllocator localAllocator, size_t bytes, size_t align)
    {
      FastAllocator::CachedAllocator* alloc = (FastAllocator::CachedAllocator*) localAllocator;
      RTC_CATCH_BEGIN;
      RTC_TRACE(rtcThreadLocalAlloc);
      return alloc->malloc0(bytes,align);
      RTC_CATCH_END(alloc->alloc->getDevice());
      return nullptr;
    }

    RTC_API void rtcMakeStaticBVH(RTCBVH hbvh)
    {
      BVH* bvh = (BVH*) hbvh;
      RTC_CATCH_BEGIN;
      RTC_TRACE(rtcStaticBVH);
      RTC_VERIFY_HANDLE(hbvh);
      bvh->morton_src.clear();
      bvh->morton_tmp.clear();
      RTC_CATCH_END(bvh->device);
    }

    RTC_API void rtcRetainBVH(RTCBVH hbvh)
    {
      BVH* bvh = (BVH*) hbvh;
      Device* device = bvh ? bvh->device : nullptr;
      RTC_CATCH_BEGIN;
      RTC_TRACE(rtcRetainBVH);
      RTC_VERIFY_HANDLE(hbvh);
      bvh->refInc();
      RTC_CATCH_END(device);
    }
    
    RTC_API void rtcReleaseBVH(RTCBVH hbvh)
    {
      BVH* bvh = (BVH*) hbvh;
      Device* device = bvh ? bvh->device : nullptr;
      RTC_CATCH_BEGIN;
      RTC_TRACE(rtcReleaseBVH);
      RTC_VERIFY_HANDLE(hbvh);
      bvh->refDec();
      RTC_CATCH_END(device);
    }


    void* createLeaf(const BVH4::NodeRef node,
                     const PrimitiveType *leafType,
                     const RTCBVHExtractFunction args,
                     void *userData) {
        size_t nb;
        if(leafType == &Triangle4v::type) {
            Triangle4v *prims = reinterpret_cast<Triangle4v *>(node.leaf(nb));
            BVHPrimitive primsArray[4*nb];
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
            BVHPrimitive primsArray[4*nb];
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
            uint geomIDs[nb];
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
            case Geometry::GTY_ROUND_HERMITE_CURVE: {
                Curve8i *curve = reinterpret_cast<Curve8i*>(prim);
                const auto N = curve->N;
                for(size_t i = 0; i < N; i++) {
                    primsArray[realNum].geomID = curve->geomID(N);
                    primsArray[realNum].primID = curve->primID(N)[i];
                    ++realNum;
                }
            } break;
            default:
                std::cout << "Unexpected geom type, which is " << ty << std::endl;
                return nullptr;
            }

            return args.createCurve(realNum, primsArray, userData);
        } else {
            std::cout << "Error: unknown prim " << leafType->name() << std::endl;
            return nullptr;
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

    void* recurse(const BVH4::NodeRef node,
                  const PrimitiveType *leafType,
                  const RTCBVHExtractFunction args,
                  void *userData) {
      if(node.isLeaf())
          return createLeaf(node, leafType, args, userData);

      const BVH4::BaseNode *bnode = nullptr;
      const BVH4::AlignedNode *anode = nullptr;
      const BVH4::AlignedNodeMB *anodeMB = nullptr;
      const BVH4::AlignedNodeMB4D *anodeMB4D = nullptr;

      const BVH4::UnalignedNode *unanode = nullptr;

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
      } else {
          std::cout << "[EMBREE - BVH] Node type is unknown -> " << node.type() << std::endl;
          return nullptr;
      }

      unsigned int nb = 0;
      void *children[4];
      for(uint i = 0; i < 4; i++) {
          void *child = recurse(bnode->child(i), leafType, args, userData);
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
              RTCAffineSpace affSpace;
              affSpace.affine[0] = unanode->naabb.p.x[i];
              affSpace.affine[1] = unanode->naabb.p.y[i];
              affSpace.affine[2] = unanode->naabb.p.z[i];

              affSpace.linear[0] = unanode->naabb.l.vx.x[i];
              affSpace.linear[1] = unanode->naabb.l.vx.y[i];
              affSpace.linear[2] = unanode->naabb.l.vx.z[i];
              affSpace.linear[3] = unanode->naabb.l.vy.x[i];
              affSpace.linear[4] = unanode->naabb.l.vy.y[i];
              affSpace.linear[5] = unanode->naabb.l.vy.z[i];
              affSpace.linear[6] = unanode->naabb.l.vz.x[i];
              affSpace.linear[7] = unanode->naabb.l.vz.y[i];
              affSpace.linear[8] = unanode->naabb.l.vz.z[i];

              args.setUnalignedBounds(child, affSpace, userData);
          }

          children[nb++] = child;
      }

      return args.createInnerNode(nb, children, userData);
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
        param.expectedSize = nullptr;

        for (Accel *a : scene->accels) {
          AccelData *ad = a->intersectors.ptr;
          if(ad->type != AccelData::TY_BVH4) {
            throw_RTCError(RTC_ERROR_INVALID_OPERATION, "Unable to extract non BVH4 tree");
            continue;
          }

          BVH4 *bvh = dynamic_cast<BVH4 *>(ad);
          BVH4::NodeRef root = bvh->root;

          recurse(root, bvh->primTy, param, &size);
        }

        args.expectedSize(size.num_prim, size.num_tri, userData);
      }

      std::vector<void*> nodes;

      for (Accel *a : scene->accels) {
        std::cout << "Accel " << a->intersectors.intersector1.name << std::endl;
        AccelData *ad = a->intersectors.ptr;
        if(ad->type != AccelData::TY_BVH4) {
          std::cout << "Unable to extract non BVH4 tree" << std::endl;
          continue;
        }

        BVH4 *bvh = dynamic_cast<BVH4 *>(ad);
        std::cout << "Prim type -> " << bvh->primTy->name() << std::endl;

        BVH4::NodeRef root = bvh->root;
        void *node = recurse(root, bvh->primTy, args, userData);
        args.setAlignedBounds(node, boundsToRTC(bvh->bounds.bounds()), userData); // TODO USe aligne bounds
        nodes.push_back(node);
      }
      std::cout << "[DONE]" << std::endl;

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

RTC_NAMESPACE_END
