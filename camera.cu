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

    const CameraData* camera = (CameraData*) optixGetSbtDataPointer();

    const unsigned int image_index = params.width * idx.y + idx.x;

    float2 subpixel_jitter = make_float2(0.5f, 0.5f) ;

    float2 d = ((make_float2(idx.x, idx.y) + subpixel_jitter) / make_float2(params.width, params.height)) * 2.f - 1.f;
    float3 ray_origin = camera->eye;
    float3 ray_direction = normalize(d.x*camera->U + d.y*camera->V + camera->W);

    float distanceMin = -1.0f; // Front face intersection distance, stays negative when missed.
    float distanceMax = -1.0f; // Back  face intersection distance, stays negative when missed.

  
    unsigned int payload = __float_as_uint(distanceMin);
    optixTrace(
        params.handle2,                     // handle
        ray_origin,                         // float3 rayOrigin
        ray_direction,                      // float3 rayDirection
        0.f,                                 // float tmin
        1e16f,                              // float tmax
        0.0f,                               // float rayTime
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
        1,                          // SBT offset (1 = CH2)
        RAY_TYPE_COUNT,             // SBT stride
        RAY_TYPE_RADIANCE,          // missSBTIndex 
        payload);


    distanceMin = __uint_as_float(payload);

    
    payload = __float_as_uint(distanceMax);
    optixTrace(
        params.handle2,                     // handle
        ray_origin,                         // float3 rayOrigin
        ray_direction,                      // float3 rayDirection
        0.f,                                 // float tmin
        1e16f,                              // float tmax
        0.0f,                               // float rayTime
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
        1,                          // SBT offset (1 = CH2)
        RAY_TYPE_COUNT,             // SBT stride
        RAY_TYPE_RADIANCE,          // missSBTIndex 
        payload);

    distanceMax = __uint_as_float(payload);



     if (0.0f < distanceMin && 0.0f < distanceMax)
        {
            if (distanceMin < distanceMax)
            {
            // The standard case: The ray started outside a volume and hit a front face and a back face farther away.
            // No special handling required. Use the two distance values as begin and end of the ray marching.
            }
            else // if distanceMin >= distanceMax
            {
            // This means a backface was hit before a father away front face.
            // In that case the ray origin must be inside a volume.
            
            distanceMin = 0.0f;
            }
        }
        else 
        {
            // Both rays missed, nothing to do here, fill per output buffer with default default data and return.
            params.frame_buffer[image_index] = make_color( make_float3(0.f) ); //why do I need this default outputbuffer???
            return;
        }

    //      else // if (0.0f < distanceMin && distanceMax < 0.0f)
    //         {
    //             // Illegal case. Front face hit but back face missed
    //             // This would mean there is no end to the volume and the ray marching would be until world end.
    //             // Maybe tint the result in some debug color to see if this happens.
    //             return;
    //         }

    float3 position;
    float3 payload_rgb = make_float3(0.f, 0.f, 0.f);
    // distanceMin = 0.f;
    // distanceMax = 10.f;
    // test transfert function min/max 
    // if alpha = 0, update distanceMin, distanceMax
    // if alpha > 0, pointQuery => look up for variance in selected range
    for (float distance = distanceMin; distance < distanceMax; distance += 0.1)
    {
        const float3 position = ray_origin + ray_direction * distance;

        optixTrace(
        params.handle,                     // handle
        position,                         // float3 rayOrigin
        ray_direction,                      // float3 rayDirection
        0.f,                                 // float tmin
        1e16f,                              // float tmax
        0.0f,                               // float rayTime
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        0,                          // SBT offset (1 = CH2)
        RAY_TYPE_COUNT,             // SBT stride
        RAY_TYPE_RADIANCE,          // missSBTIndex 
        float3_as_args(payload_rgb));
    }


    
    params.frame_buffer[image_index] = make_color( payload_rgb );
  

}





// extern "C" __global__ void __raygen__pinhole_camera()
// {
//     const uint3 idx = optixGetLaunchIndex();

//     const CameraData* camera = (CameraData*) optixGetSbtDataPointer();

//     const unsigned int image_index = params.width * idx.y + idx.x;

//     float2 subpixel_jitter = make_float2(0.5f, 0.5f) ;

//     float2 d = ((make_float2(idx.x, idx.y) + subpixel_jitter) / make_float2(params.width, params.height)) * 2.f - 1.f;
//     float3 ray_origin = camera->eye;
//     float3 ray_direction = normalize(d.x*camera->U + d.y*camera->V + camera->W);

//     float3 position;
//     float3 payload_rgb = make_float3(0.f, 0.f, 0.f);
  
//     for (float distance = 0.f; distance < 20.f; distance += 0.05)
//     {
//         const float3 position = ray_origin + ray_direction * distance;

//         optixTrace(
//         params.handle,                     // handle
//         position,                         // float3 rayOrigin
//         ray_direction,                      // float3 rayDirection
//         0.f,                                 // float tmin
//         1e16f,                              // float tmax
//         0.0f,                               // float rayTime
//         OptixVisibilityMask( 1 ),
//         OPTIX_RAY_FLAG_NONE,
//         0,                          // SBT offset (1 = CH2)
//         RAY_TYPE_COUNT,             // SBT stride
//         RAY_TYPE_RADIANCE,          // missSBTIndex 
//         float3_as_args(payload_rgb));
//     }


    
//     params.frame_buffer[image_index] = make_color( payload_rgb );
  

// }