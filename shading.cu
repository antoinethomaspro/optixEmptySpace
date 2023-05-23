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

static __device__ __inline__ OcclusionPRD getOcclusionPRD()
{
    OcclusionPRD prd;
    prd.attenuation.x = int_as_float( optixGetPayload_0() );
    prd.attenuation.y = int_as_float( optixGetPayload_1() );
    prd.attenuation.z = int_as_float( optixGetPayload_2() );
    return prd;
}

static __device__ __inline__ void setOcclusionPRD( const OcclusionPRD &prd )
{
    optixSetPayload_0( float_as_int(prd.attenuation.x) );
    optixSetPayload_1( float_as_int(prd.attenuation.y) );
    optixSetPayload_2( float_as_int(prd.attenuation.z) );
}

static __device__ __inline__ float3
traceRadianceRay(
    float3 origin,
    float3 direction,
    int depth,
    float importance)
{
    RadiancePRD prd;
    prd.depth = depth;
    prd.importance = importance;

    optixTrace(
        params.handle,
        origin,
        direction,
        params.scene_epsilon,
        1e16f,
        0.0f,
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,
        RAY_TYPE_COUNT,
        RAY_TYPE_RADIANCE,
        float3_as_args(prd.result),
        /* Can't use float_as_int() because it returns rvalue but payload requires a lvalue */
        reinterpret_cast<unsigned int&>(prd.importance),
        reinterpret_cast<unsigned int&>(prd.depth) );

    return prd.result;
}



extern "C" __global__ void __closesthit__metal_radiance()
{
    RadiancePRD prd;
    
    prd.result = make_float3(1.f, 0.f, 0.f);
    setRadiancePRD(prd);
}

extern "C" __global__ void __miss__constant_bg()
{
    const MissData* sbt_data = (MissData*) optixGetSbtDataPointer();
    RadiancePRD prd = getRadiancePRD();
    prd.result = sbt_data->bg_color;
    setRadiancePRD(prd);
}
