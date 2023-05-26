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

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>
#include <iomanip>
#include <cstring>

#include "optixWhitted.h"
#include "CUDABuffer.h"

 






//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------


bool              resize_dirty  = false;
bool              minimized     = false;

// Camera state
bool              camera_changed = true;
sutil::Camera     camera;
sutil::Trackball  trackball;

// Mouse state
int32_t           mouse_button = -1;

const int         max_trace = 12;

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT )

    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<CameraData>      RayGenRecord;
typedef Record<MissData>        MissRecord;
typedef Record<HitGroupData>   HitGroupSbtRecord;


struct WhittedState
{
    OptixDeviceContext          context                   = 0;
    OptixTraversableHandle      gas_handle                = {};
    CUdeviceptr                 d_gas_output_buffer       = {};


    OptixModule                 camera_module             = 0;
    OptixModule                 shading_module            = 0;

    OptixProgramGroup           raygen_prog_group         = 0;
    OptixProgramGroup           radiance_miss_prog_group  = 0;
    OptixProgramGroup           occlusion_miss_prog_group = 0;
    OptixProgramGroup           radiance_metal_sphere_prog_group  = 0;
    OptixProgramGroup           mesh_hit_prog_group = 0;
    OptixProgramGroup           mesh_hit_prog_group2 = 0;

   

    OptixPipeline               pipeline                  = 0;
    OptixPipelineCompileOptions pipeline_compile_options  = {};

    CUstream                    stream                    = 0;
    Params                      params;
    Params*                     d_params                  = nullptr;

    OptixShaderBindingTable     sbt                       = {};
};

//------------------------------------------------------------------------------
//
//  Geometry and Camera data
//
//------------------------------------------------------------------------------

// Metal sphere, glass sphere, floor, light
const Sphere g_sphere = {
    { 2.0f, 1.5f, -2.5f }, // center
    1.0f                   // radius
};


//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );

    if( action == GLFW_PRESS )
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>( xpos ), static_cast<int>( ypos ));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );

    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        camera_changed = true;
    }
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );
    params->width  = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty   = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q ||
            key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if(trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}

void initLaunchParams( WhittedState& state )
{
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.params.accum_buffer ),
        state.params.width*state.params.height*sizeof(float4)
    ) );
    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    state.params.subframe_index = 0u;


    state.params.max_depth = max_trace;
    state.params.scene_epsilon = 1.e-4f;

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );

    state.params.handle = state.gas_handle;
}

static void buildGas(
    const WhittedState &state,
    const OptixAccelBuildOptions &accel_options,
    const OptixBuildInput &build_input,
    OptixTraversableHandle &gas_handle,
    CUdeviceptr &d_gas_output_buffer
    )
{
    OptixAccelBufferSizes gas_buffer_sizes;
    CUdeviceptr d_temp_buffer_gas;

    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes));

    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_gas ),
        gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output and size of compacted GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                compactedSizeOffset + 8
                ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK( optixAccelBuild(
        state.context,
        0,
        &accel_options,
        &build_input,
        1,
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        &emitProperty,
        1) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

static void sphere_bound(float3 center, float radius, float result[6])
{
    OptixAabb *aabb = reinterpret_cast<OptixAabb*>(result);

    float3 m_min = center - radius;
    float3 m_max = center + radius;

    *aabb = {
        m_min.x, m_min.y, m_min.z,
        m_max.x, m_max.y, m_max.z
    };
}

void createGeometry( WhittedState &state )
{
    //
    // Build Custom Primitives
    //
    const Sphere g_sphere = {
    { -13.0f, 9.5f, -5.5f }, // center
    3.0f                   // radius
        };

    // Load AABB into device memory
    OptixAabb   aabb[1];
    CUdeviceptr d_aabb;

    sphere_bound(
        g_sphere.center, g_sphere.radius,
        reinterpret_cast<float*>(&aabb[0]));


    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb
        ), 1 * sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_aabb ),
                &aabb,
                1 * sizeof( OptixAabb ),
                cudaMemcpyHostToDevice
                ) );

    // Setup AABB build input
    uint32_t aabb_input_flags[] = {
        /* flags for metal sphere */
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        /* flag for glass sphere */
        OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL,
        /* flag for floor */
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    };
    /* TODO: This API cannot control flags for different ray type */

    const uint32_t sbt_index[] = { 0 };
    CUdeviceptr    d_sbt_index;

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_index ), sizeof(sbt_index) ) );
    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>( d_sbt_index ),
        sbt_index,
        sizeof( sbt_index ),
        cudaMemcpyHostToDevice ) );

    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb;
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = 1;
    aabb_input.customPrimitiveArray.numPrimitives = 1;
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer         = d_sbt_index;
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes    = sizeof( uint32_t );
    aabb_input.customPrimitiveArray.primitiveIndexOffset         = 0;


    OptixAccelBuildOptions accel_options = {
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION,  // buildFlags
        OPTIX_BUILD_OPERATION_BUILD         // operation
    };


    buildGas(
        state,
        accel_options,
        aabb_input,
        state.gas_handle,
        state.d_gas_output_buffer);

    CUDA_CHECK( cudaFree( (void*)d_aabb) );
}





static void buildTriangle(const WhittedState &state, OptixTraversableHandle &gas_handle)
{
    std::array<float3, 3> arr = {{
    { -1.5f, -2.5f, 2.0f },
    {  3.5f, -20.5f, 3.0f},
    {  1.0f,  1.5f, 1.0f }
        }};

    std::array<float3, 3> arr2 = { {
    { -0.5f, -0.5f, 0.0f },
    {  0.5f, -0.5f, 0.0f },
    {  0.0f,  0.5f, 0.0f }
            } };

    class TriangleMesh {
    public:
    std::vector<float3> vertex;
    std::vector<int3> index;
        };

    TriangleMesh mesh1;
    mesh1.vertex.insert(mesh1.vertex.end(), arr.begin(), arr.end());

    TriangleMesh mesh2;
    mesh2.vertex.insert(mesh2.vertex.end(), arr2.begin(), arr2.end());

    /*! the model we are going to trace rays against */
    std::vector<TriangleMesh> meshes;
    /*! one buffer per input mesh */
    std::vector<CUDABuffer> vertexBuffer;

    meshes.push_back(mesh1);
    meshes.push_back(mesh2);


    vertexBuffer.resize(meshes.size());
    
    OptixTraversableHandle asHandle { 0 };

    // ==================================================================
    // triangle inputs
    // ==================================================================
	std::vector<OptixBuildInput> triangleInput(meshes.size());
    std::vector<CUdeviceptr> d_vertices(meshes.size());
	// std::vector<CUdeviceptr> d_indices(meshes.size());
	std::vector<uint32_t> triangleInputFlags(meshes.size());

    for (int meshID=0;meshID<meshes.size();meshID++) {
    // upload the model to the device: the builder
    TriangleMesh &model = meshes[meshID];
    vertexBuffer[meshID].alloc_and_upload(model.vertex);
    // indexBuffer[meshID].alloc_and_upload(model.index);

    triangleInput[meshID] = {};
    triangleInput[meshID].type
      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
    // d_indices[meshID]  = indexBuffer[meshID].d_pointer();
      
    triangleInput[meshID].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(float3);
    triangleInput[meshID].triangleArray.numVertices         = (int)model.vertex.size();
    triangleInput[meshID].triangleArray.vertexBuffers       = &d_vertices[meshID];
    
    // triangleInput[meshID].triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    // triangleInput[meshID].triangleArray.indexStrideInBytes  = sizeof(vec3i);
    // triangleInput[meshID].triangleArray.numIndexTriplets    = (int)model.index.size();
    // triangleInput[meshID].triangleArray.indexBuffer         = d_indices[meshID];
    
    triangleInputFlags[meshID] = 0 ;
    
    // in this example we have one SBT entry, and no per-primitive
    // materials:
    triangleInput[meshID].triangleArray.flags               = &triangleInputFlags[meshID];
    triangleInput[meshID].triangleArray.numSbtRecords               = 1;
    triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer        = 0; 
    triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes   = 0; 
    triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0; 
    }

    // ==================================================================
    // BLAS setup
    // ==================================================================
    
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys  = 1;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                ( state.context,
                 &accelOptions,
                 triangleInput.data(),
                 (int)meshes.size(),  // num_build_inputs
                 &blasBufferSizes
                 ));

    // ==================================================================
    // prepare compaction
    // ==================================================================
    
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));
    
    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();


    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    
    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    
    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
      
    OPTIX_CHECK(optixAccelBuild(state.context,
                                /* stream */0,
                                &accelOptions,
                                triangleInput.data(),
                                (int)meshes.size(),  
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,
                                
                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,
                                
                                &gas_handle,
                                
                                &emitDesc,1
                                ));
    CUDA_SYNC_CHECK();

     // ==================================================================
    // perform compaction ????
    // ==================================================================
   

}




void createModules( WhittedState &state )
{
    OptixModuleCompileOptions module_compile_options = {};

    char log[2048];
    size_t sizeof_log = sizeof(log);

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "camera.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            log,
            &sizeof_log,
            &state.camera_module ) );
    }

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "shading.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            log,
            &sizeof_log,
            &state.shading_module ) );
    }

  
}

static void createCameraProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           cam_prog_group;
    OptixProgramGroupOptions    cam_prog_group_options = {};
    OptixProgramGroupDesc       cam_prog_group_desc = {};
    cam_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cam_prog_group_desc.raygen.module = state.camera_module;
    cam_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole_camera";

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &cam_prog_group_desc,
        1,
        &cam_prog_group_options,
        log,
        &sizeof_log,
        &cam_prog_group ) );

    program_groups.push_back(cam_prog_group);
    state.raygen_prog_group = cam_prog_group;
}



static void createMetalSphereProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_metal_sphere_prog_group;
    OptixProgramGroupOptions    radiance_metal_sphere_prog_group_options = {};
    OptixProgramGroupDesc       radiance_metal_sphere_prog_group_desc = {};
    radiance_metal_sphere_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        radiance_metal_sphere_prog_group_desc.hitgroup.moduleIS           = state.shading_module;
    radiance_metal_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere";
    radiance_metal_sphere_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_metal_sphere_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__mesh";
    radiance_metal_sphere_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_metal_sphere_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_metal_sphere_prog_group_desc,
        1,
        &radiance_metal_sphere_prog_group_options,
        log,
        &sizeof_log,
        &radiance_metal_sphere_prog_group ) );

    program_groups.push_back(radiance_metal_sphere_prog_group);
    state.radiance_metal_sphere_prog_group = radiance_metal_sphere_prog_group;  
}


static void createMeshProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           mesh_hit_prog_group;
    OptixProgramGroupOptions    mesh_hit_prog_group_options = {};
    OptixProgramGroupDesc       mesh_hit_prog_group_desc = {};
    mesh_hit_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
    mesh_hit_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    mesh_hit_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__mesh";
    mesh_hit_prog_group_desc.hitgroup.moduleIS               = nullptr;
    mesh_hit_prog_group_desc.hitgroup.entryFunctionNameIS    = nullptr;
    mesh_hit_prog_group_desc.hitgroup.moduleAH               = nullptr;
    mesh_hit_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

     char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &mesh_hit_prog_group_desc,
        1,
        &mesh_hit_prog_group_options,
        log,
        &sizeof_log,
        &mesh_hit_prog_group ) );

    program_groups.push_back(mesh_hit_prog_group);
    state.mesh_hit_prog_group = mesh_hit_prog_group;  
}

static void createMeshProgram2( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           mesh_hit_prog_group2;
    OptixProgramGroupOptions    mesh_hit_prog_group2_options = {};
    OptixProgramGroupDesc       mesh_hit_prog_group2_desc = {};
    mesh_hit_prog_group2_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
    mesh_hit_prog_group2_desc.hitgroup.moduleCH               = state.shading_module;
    mesh_hit_prog_group2_desc.hitgroup.entryFunctionNameCH    = "__closesthit__mesh2";
    mesh_hit_prog_group2_desc.hitgroup.moduleIS               = nullptr;
    mesh_hit_prog_group2_desc.hitgroup.entryFunctionNameIS    = nullptr;
    mesh_hit_prog_group2_desc.hitgroup.moduleAH               = nullptr;
    mesh_hit_prog_group2_desc.hitgroup.entryFunctionNameAH    = nullptr;

     char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &mesh_hit_prog_group2_desc,
        1,
        &mesh_hit_prog_group2_options,
        log,
        &sizeof_log,
        &mesh_hit_prog_group2 ) );

    program_groups.push_back(mesh_hit_prog_group2);
    state.mesh_hit_prog_group2 = mesh_hit_prog_group2;  
}


static void createMissProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroupOptions    miss_prog_group_options = {};
    OptixProgramGroupDesc       miss_prog_group_desc = {};
    miss_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module             = state.shading_module;
    miss_prog_group_desc.miss.entryFunctionName  = "__miss__constant_bg";

    char    log[2048];
    size_t  sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        log,
        &sizeof_log,
        &state.radiance_miss_prog_group ) );

    program_groups.push_back(state.radiance_miss_prog_group);

    miss_prog_group_desc.miss = {
        nullptr,    // module
        nullptr     // entryFunctionName
    };
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        log,
        &sizeof_log,
        &state.occlusion_miss_prog_group ) );

    program_groups.push_back(state.occlusion_miss_prog_group);
}

void createPipeline( WhittedState &state )
{
    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_compile_options = {
        false,                                                  // usesMotionBlur
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags
        5,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues
        5,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues
        OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
        "params"                                                // pipelineLaunchParamsVariableName
    };

    // Prepare program groups
    createModules( state );
    createCameraProgram( state, program_groups );
    createMetalSphereProgram( state, program_groups );
    createMeshProgram( state, program_groups );
    createMeshProgram2( state, program_groups );
    


   // createMetalSphereProgram( state, program_groups );
    createMissProgram( state, program_groups );

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {
        max_trace,                          // maxTraceDepth
        OPTIX_COMPILE_DEBUG_LEVEL_FULL      // debugLevel
    };
    char    log[2048];
    size_t  sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG( optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        static_cast<unsigned int>( program_groups.size() ),
        log,
        &sizeof_log,
        &state.pipeline ) );

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace,
                                             0,  // maxCCDepth
                                             0,  // maxDCDepth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
}

void syncCameraDataToSbt( WhittedState &state, const CameraData& camData )
{
    RayGenRecord rg_sbt;

    optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt );
    rg_sbt.data = camData;

    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>( state.sbt.raygenRecord ),
        &rg_sbt,
        sizeof( RayGenRecord ),
        cudaMemcpyHostToDevice
    ) );
}

void createSBT( WhittedState &state )
{
    // Raygen program record
    {
        CUdeviceptr d_raygen_record;
        size_t sizeof_raygen_record = sizeof( RayGenRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_raygen_record ),
            sizeof_raygen_record ) );

        state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        CUdeviceptr d_miss_record;
        size_t sizeof_miss_record = sizeof( MissRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_miss_record ),
            sizeof_miss_record*RAY_TYPE_COUNT ) );

        MissRecord ms_sbt[RAY_TYPE_COUNT];
        optixSbtRecordPackHeader( state.radiance_miss_prog_group, &ms_sbt[0] );
        optixSbtRecordPackHeader( state.occlusion_miss_prog_group, &ms_sbt[1] );
        ms_sbt[1].data = ms_sbt[0].data = { 0.34f, 0.55f, 0.85f };

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_miss_record ),
            ms_sbt,
            sizeof_miss_record*RAY_TYPE_COUNT,
            cudaMemcpyHostToDevice
        ) );

        state.sbt.missRecordBase          = d_miss_record;
        state.sbt.missRecordCount         = RAY_TYPE_COUNT;
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof_miss_record );
    }

    // Hitgroup program record

     
    {

        // CUDABuffer hitgroupRecordsBuffer;

        // std::vector<HitGroupSbtRecord> hitgroupRecords;

        // HitGroupSbtRecord rec1;
        // OPTIX_CHECK(optixSbtRecordPackHeader(state.mesh_hit_prog_group, &rec1));
        // hitgroupRecords.push_back(rec1);

        // HitGroupSbtRecord rec2;
        // OPTIX_CHECK(optixSbtRecordPackHeader(state.mesh_hit_prog_group2, &rec2));
        // hitgroupRecords.push_back(rec2);

        // hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);

        // state.sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
        // state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        // state.sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();







        CUdeviceptr hitgroup_record;
        size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
        HitGroupSbtRecord hg_sbt;


        OPTIX_CHECK( optixSbtRecordPackHeader( state.radiance_metal_sphere_prog_group, &hg_sbt ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( hitgroup_record ),
                    &hg_sbt,
                    hitgroup_record_size,
                    cudaMemcpyHostToDevice
                    ) );

        state.sbt.hitgroupRecordBase            = hitgroup_record;  //ok
        state.sbt.hitgroupRecordStrideInBytes   = sizeof( HitGroupSbtRecord );
        state.sbt.hitgroupRecordCount           = 1;

    }
    
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

void createContext( WhittedState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    state.context = context;
}

//
//
//

void initCameraState()
{
    camera.setEye( make_float3( 8.0f, 2.0f, -4.0f ) );
    camera.setLookat( make_float3( 4.0f, 2.3f, -4.0f ) );
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 60.0f );
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    trackball.setGimbalLock(true);
}

void handleCameraUpdate( WhittedState &state )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( state.params.width ) / static_cast<float>( state.params.height ) );
    CameraData camData;
    camData.eye = camera.eye();
    camera.UVWFrame( camData.U, camData.V, camData.W );

    syncCameraDataToSbt(state, camData);
}

void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &params.accum_buffer ),
        params.width*params.height*sizeof(float4)
    ) );
}

void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, WhittedState &state )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        state.params.subframe_index = 0;

    handleCameraUpdate( state );
    handleResize( output_buffer, state.params );
}

void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, WhittedState& state )
{

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( state.d_params ),
                                 &state.params,
                                 sizeof( Params ),
                                 cudaMemcpyHostToDevice,
                                 state.stream
    ) );

    OPTIX_CHECK( optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>( state.d_params ),
        sizeof( Params ),
        &state.sbt,
        state.params.width,  // launch width
        state.params.height, // launch height
        1                    // launch depth
    ) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}


void displaySubframe(
    sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
    sutil::GLDisplay&                 gl_display,
    GLFWwindow*                       window )
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}


void cleanupState( WhittedState& state )
{
    OPTIX_CHECK( optixPipelineDestroy     ( state.pipeline                ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.raygen_prog_group       ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_metal_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_miss_prog_group         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.shading_module          ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.camera_module           ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context                 ) );


    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer    ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.accum_buffer    ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params               ) ) );
}

int main( int argc, char* argv[] )
{
    WhittedState state;
    state.params.width  = 768;
    state.params.height = 768;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
    std::string outfile;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--no-gl-interop" )
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            outfile = argv[++i];
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            int w, h;
            sutil::parseDimensions( dims_arg.c_str(), w, h );
            state.params.width  = w;
            state.params.height = h;
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        initCameraState();

        //
        // Set up OptiX state
        //
        createContext  ( state );

       // buildMesh( state, state.gas_handle, state.d_gas_output_buffer);

        //buildTriangle(state, state.gas_handle);
        
        createGeometry  ( state );

  
        createPipeline ( state );
        createSBT      ( state );

        initLaunchParams( state );

        //
        // Render loop
        //
        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "optixWhitted", state.params.width, state.params.height );
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );
            glfwSetScrollCallback       ( window, scrollCallback        );
            glfwSetWindowUserPointer    ( window, &state.params         );

            {
                // output_buffer needs to be destroyed before cleanupUI is called
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                        output_buffer_type,
                        state.params.width,
                        state.params.height
                        );

                output_buffer.setStream( state.stream );
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState( output_buffer, state );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe( output_buffer, state );
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe( output_buffer, gl_display, window );
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats( state_update_time, render_time, display_time );

                    glfwSwapBuffers( window );

                    ++state.params.subframe_index;
                }
                while( !glfwWindowShouldClose( window ) );

            }
            sutil::cleanupUI( window );
        }
        else
        {
            if ( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                sutil::initGLFW(); // For GL context
                sutil::initGL();
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                    );

            handleCameraUpdate( state );
            handleResize( output_buffer, state.params );
            launchSubframe( output_buffer, state );

            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = output_buffer.width();
            buffer.height       = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            sutil::saveImage( outfile.c_str(), buffer, false );

            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                glfwTerminate();
            }
        }

        cleanupState( state );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
