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

#include <cmath>

#include <vector_types.h>
#include <optix_device.h>

#include "optixWhitted.h"
#include "helpers.h"

extern "C" {
__constant__ Params params;
}

#define float3_as_ints( u ) float_as_int( u.x ), float_as_int( u.y ), float_as_int( u.z )




static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
}


static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
            int_as_float( optixGetPayload_0() ),
            int_as_float( optixGetPayload_1() ),
            int_as_float( optixGetPayload_2() )
            );
}


extern "C" __global__ void __miss__constant_bg()
{

}

extern "C" __global__ void __closesthit__mesh()
{
    // float3  payload = getPayload();

    // const int primID = optixGetPrimitiveIndex();

    // const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

    // const Face face = hit_group_data->face[primID];

    // int2 elemIDs = face.elemIDs;

    // int elemID = optixIsTriangleBackFaceHit() ? elemIDs.x : elemIDs.y;
    // if (elemID < 0) return;
    

    //  switch(elemID){
    //     case 0:
    //         setPayload( payload + make_float3( 0.1f, 0.f, 0.f));
    //         break;
    //     case 1:
    //         setPayload( payload + make_float3( 0.f, 0.1f, 0.f));
    //         break;
    //     case 2:
    //         setPayload( payload + make_float3( 0.f, 0.f, 0.1f));
    //         break;
    //     case 3:
    //         setPayload( payload + make_float3( 1.f, 1.f, 0.f));
    //         break;
    //     case 4:
    //         setPayload( payload + make_float3( 0.f, 1.f, 1.f));
    //         break;
    //  }

    float3  payload = getPayload();
    const int primID = optixGetPrimitiveIndex();

    if(optixIsTriangleBackFaceHit() ==true)
        {  setPayload( payload + make_float3( 0.1f, 0.f, 0.f));}
        else{return;}

  
}


extern "C" __global__ void __closesthit__mesh2() //handle primary rays
{

    // float3  payload = getPayload();

    // if(optixIsTriangleBackFaceHit() ==true)
    //     {  setPayload( payload + make_float3( 0.f, 0.1f, 0.f));}
    //     else{return;}

    float tmax = optixGetRayTmax ();
    int a = 2;

    optixSetPayload_0( float_as_int( tmax ) );
    optixSetPayload_1(a);
    

    
}