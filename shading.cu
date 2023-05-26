//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <vector_types.h>
#include <optix_device.h>

#include "optixWhitted.h"
#include "helpers.h"

extern "C" {
__constant__ Params params;
}

#define float3_as_ints( u ) float_as_int( u.x ), float_as_int( u.y ), float_as_int( u.z )

extern "C" __global__ void __intersection__sphere()
{
    //const SphereHitGroupData* hit_group_data = reinterpret_cast<SphereHitGroupData*>( optixGetSbtDataPointer() );

    const int primID = optixGetPrimitiveIndex();

    optixReportIntersection(0., 0. , primID);
}

static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
            int_as_float( optixGetPayload_0() ),
            int_as_float( optixGetPayload_1() ),
            int_as_float( optixGetPayload_2() )
            );
}

static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
}

static __device__ __inline__ RadiancePRD getRadiancePRD()
{
    RadiancePRD prd;
    prd.result.x = int_as_float( optixGetPayload_0() );
    prd.result.y = int_as_float( optixGetPayload_1() );
    prd.result.z = int_as_float( optixGetPayload_2() );
    prd.importance = int_as_float( optixGetPayload_3() );
    prd.depth = optixGetPayload_4();
    return prd;
}

static __device__ __inline__ void setRadiancePRD( const RadiancePRD &prd )
{
    optixSetPayload_0( float_as_int(prd.result.x) );
    optixSetPayload_1( float_as_int(prd.result.y) );
    optixSetPayload_2( float_as_int(prd.result.z) );
    optixSetPayload_3( float_as_int(prd.importance) );
    optixSetPayload_4( prd.depth );
}





extern "C" __global__ void __closesthit__mesh2()
{
    setPayload( make_float3(1.f, 0.f, 0.f));
}

extern "C" __global__ void __miss__constant_bg()
{
    const MissData* sbt_data = (MissData*) optixGetSbtDataPointer();
    RadiancePRD prd = getRadiancePRD();
    prd.result = sbt_data->bg_color;
    setRadiancePRD(prd);
}

extern "C" __global__ void __closesthit__mesh()
{
    setPayload( make_float3(0.f, 1.f, 0.f));
}

extern "C" __global__ void __closesthit__ch()
{
    float3  payload = getPayload();
    int primID = optixGetAttribute_0();

    switch(primID){
        case 0:
            setPayload( payload + make_float3( 1.f, 0.f, 0.f));
            break;
        case 1:
            setPayload( payload + make_float3( 0.f, 0.05f, 0.f));
            break;
        case 2:
            setPayload( payload + make_float3( 0.f, 0.f, 0.05f));
            break;
        case 3:
            setPayload( payload + make_float3( 0.05, 0.05f, 0.f));
            break;
        case 4:
            setPayload( payload + make_float3( 0.f, 0.05f, 0.05f));
            break;
    }
    
}