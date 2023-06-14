extern "C" __global__ void __raygen__volume()
{
 // Calculate primary ray orgin and direction here
 ...
 float3 origin    = launchParams.cam_eye;
 float3 direction = normalize(launchParams.camU * d.x + 
                              launchParams.camV * d.y + 
                              launchParams.camW);

  // This code block needs to be inside a loop which advances the the ray orgin or tmin value to the next AABB entry point until both rays miss.

  float distanceMin = -1.0f; // Front face intersection distance, stays negative when missed.
  float distanceMax = -1.0f; // Back  face intersection distance, stays negative when missed.

  // Shoot primary ray with back face culling enabled.
  unsigned int payload = __float_as_uint(distanceMin);
  optixTrace(launchParams.topObject,
             origin, direction, // origin, direction
             0.0f, RT_DEFAULT_MAX, 0.0f, // tmin, tmax, time
             OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
             TYPE_RAY_PROBE_VOLUME, NUM_RAY_TYPES, TYPE_RAY_PROBE_VOLUME,
             payload);
  distanceMin = __uint_as_float(payload);

  // Shoot primary ray with front face culling enabled.
  payload = __float_as_uint(distanceMax);
  optixTrace(launchParams.topObject,
             origin, direction, // origin, direction
             0.0f, RT_DEFUALT_MAX, 0.0f, // tmin, tmax, time
             OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
             TYPE_RAY_PROBE_VOLUME, NUM_RAY_TYPES, TYPE_RAY_PROBE_VOLUME,
             payload);
  distanceMax = __uint_as_float(payload);
 
  // There are four possible case here.
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
      distanceMax = distanceMin; // to be checked. 
      distanceMin = 0.0f;
    }
  }
  else  if (distanceMin < 0.0f && distanceMax < 0.0f)
  {
    // Both rays missed, nothing to do here, fill per output buffer with default default data and return.
  }
  else if (distanceMin < 0.0f && 0.0f < distanceMax)
  {
    // Primary ray origin is inside some volume. 
    distanceMin = 0.0f; // Start ray march from origin.
  }
  else // if (0.0f < distanceMin && distanceMax < 0.0f)
  {
    // Illegal case. Front face hit but back face missed
    // This would mean there is no end to the volume and the ray marching would be until world end.
    // Maybe tint the result in some debug color to see if this happens.
  }
 
  float3 result = make_float3(0.0f);
  
  for (float distance = distanceMin; distance < distanceMax; distance += STEP_DISTANCE)
  {
    // Calculate world position of the volume sample point.
    const float3 position = origin + direction * distance;
    
    // Somehow calculate the 3D volume data element (e.g. a 3D texture coordinate in normalized range [0.0f, 1.0f]^3
    const float3 sample = getVolumeSampleCoordinate(position);
    
    // Read some volume data from the sample position.
    const float3 data = some_volume_data_lookup(sample);
    
    // Accumulate to the result you need.
    result += some_operation(data); 
  }
  
  // Write the per launch index result to your output buffer.
  ...
}