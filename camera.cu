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
#include "random.h"
#include "helpers.h"
#include <cuda/helpers.h>


extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__pinhole_camera()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const CameraData* camera = (CameraData*) optixGetSbtDataPointer();

    const unsigned int image_index = params.width * idx.y + idx.x;
    unsigned int       seed        = tea<16>( image_index, params.subframe_index );

    // Subpixel jitter: send the ray through a different position inside the pixel each time,
    // to provide antialiasing. The center of each pixel is at fraction (0.5,0.5)
    float2 subpixel_jitter = params.subframe_index == 0 ?
        make_float2(0.5f, 0.5f) : make_float2(rnd( seed ), rnd( seed ));

    float2 d = ((make_float2(idx.x, idx.y) + subpixel_jitter) / make_float2(params.width, params.height)) * 2.f - 1.f;
    float3 ray_origin = camera->eye;
    float3 ray_direction = normalize(d.x*camera->U + d.y*camera->V + camera->W);

    RadiancePRD prd;
    prd.importance = 1.f;
    prd.depth = 0;

    float3 payload_rgb = make_float3( 0.f, 0.f, 0.f);

    //OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
    //OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,

    
    for (float i = 0.f; i < 10.f; i+=0.1f ) {

    float3 origin = ray_origin + ray_direction*i;

    optixTrace(
        params.handle2,
        origin,
        ray_direction,
        0.,
        1e16f,
        0.0f,
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
       // OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
       // OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
        1,                          // SBT offset
        RAY_TYPE_COUNT,             // SBT stride
        RAY_TYPE_RADIANCE,          // missSBTIndex 
        float3_as_args(payload_rgb));

    }

    float3 res = payload_rgb ;

    // for (float i = 0.f; i < 10.f; i+=0.1f ) {

    // float3 origin = ray_origin + ray_direction*i;

    // optixTrace(
    //     params.handle2,
    //     origin,
    //     ray_direction,
    //     0.,
    //     1e16f,
    //     0.0f,
    //     OptixVisibilityMask( 1 ),
    //     OPTIX_RAY_FLAG_NONE,
    //    // OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
    //    // OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
    //     0,                          // SBT offset                      
    //     RAY_TYPE_COUNT,             // SBT stride             
    //     RAY_TYPE_RADIANCE,          // missSBTIndex
    //     float3_as_args(payload_rgb));

    // }

    //float3 res2 = payload_rgb ;

    params.frame_buffer[image_index] = make_color( res );

}



// extern "C" __global__ void __raygen__pinhole_camera()
// {
//     const uint3 idx = optixGetLaunchIndex();
//     const uint3 dim = optixGetLaunchDimensions();

//     const CameraData* camera = (CameraData*) optixGetSbtDataPointer();

//     const unsigned int image_index = params.width * idx.y + idx.x;
//     unsigned int       seed        = tea<16>( image_index, params.subframe_index );

//     // Subpixel jitter: send the ray through a different position inside the pixel each time,
//     // to provide antialiasing. The center of each pixel is at fraction (0.5,0.5)
//     float2 subpixel_jitter = params.subframe_index == 0 ?
//         make_float2(0.5f, 0.5f) : make_float2(rnd( seed ), rnd( seed ));

//     float2 d = ((make_float2(idx.x, idx.y) + subpixel_jitter) / make_float2(params.width, params.height)) * 2.f - 1.f;
//     float3 ray_origin = camera->eye;
//     float3 ray_direction = normalize(d.x*camera->U + d.y*camera->V + camera->W);


//   // This code block needs to be inside a loop which advances the the ray orgin or tmin value to the next AABB entry point until both rays miss.

//   float distanceMin = -1.0f; // Front face intersection distance, stays negative when missed.
//   float distanceMax = -1.0f; // Back  face intersection distance, stays negative when missed.

//   // Shoot primary ray with back face culling enabled.
//   unsigned int payload = __float_as_uint(distanceMin);
//   optixTrace(params.handle,
//              ray_origin, ray_direction, // origin, direction
//              0.0f, 1e16f, 0.0f, // tmin, tmax, time
//              OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
//              RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE,
//              payload);
//   distanceMin = __uint_as_float(payload);

//   // Shoot primary ray with front face culling enabled.
//   payload = __float_as_uint(distanceMax);
//   optixTrace(params.handle,
//              ray_origin, ray_direction, // origin, direction
//              0.0f, 1e16f, 0.0f, // tmin, tmax, time
//              OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
//              RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE,
//              payload);
//   distanceMax = __uint_as_float(payload);
 
//   // There are four possible case here.
//   if (0.0f < distanceMin && 0.0f < distanceMax)
//   {
//     if (distanceMin < distanceMax)
//     {
//       // The standard case: The ray started outside a volume and hit a front face and a back face farther away.
//       // No special handling required. Use the two distance values as begin and end of the ray marching.
//     }
//     else // if distanceMin >= distanceMax
//     {
//       // This means a backface was hit before a father away front face.
//       // In that case the ray origin must be inside a volume.
//       distanceMin = 0.0f;
//     }
//   }
//   else  if (distanceMin < 0.0f && distanceMax < 0.0f)
//   {
//     // Both rays missed, nothing to do here, fill per output buffer with default default data and return.
//       float3 result = make_float3(0.0f);
//       params.frame_buffer[image_index] = make_color( result );
//   }
//   else if (distanceMin < 0.0f && 0.0f < distanceMax)
//   {
//     // Primary ray origin is inside some volume. 
//     distanceMin = 0.0f; // Start ray march from origin.
//   }
//   else // if (0.0f < distanceMin && distanceMax < 0.0f)
//   {
//     // Illegal case. Front face hit but back face missed
//     // This would mean there is no end to the volume and the ray marching would be until world end.
//     // Maybe tint the result in some debug color to see if this happens.
//   }
 
//   float3 payload_rgb = make_float3( 0.f, 0.f, 0.f);
  
//   for (float distance = distanceMin; distance < distanceMax; distance += 0.05)
//   {
//     // Calculate world position of the volume sample point.
//     const float3 position = ray_direction * distance;

//      optixTrace(
//         params.handle,
//         position,
//         ray_direction,
//         0.,
//         1e16f,
//         0.0f,
//         OptixVisibilityMask( 1 ),
//         OPTIX_RAY_FLAG_NONE,
//         RAY_TYPE_RADIANCE,
//         RAY_TYPE_COUNT,
//         RAY_TYPE_RADIANCE,
//         float3_as_args(payload_rgb));

//   }
  
//   // Write the per launch index result to your output buffer.
//     params.frame_buffer[image_index] = make_color( payload_rgb );

// }