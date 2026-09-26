add_library(ninfer_runtime_support STATIC
  engine/admission_policy.cpp
  engine/context_cache/context_cost.cpp
  engine/context_cache/context_cost_defaults.cpp
  engine/kv_capacity.cpp
  engine/public_types.cpp
)
ninfer_internal_includes(ninfer_runtime_support)
target_link_libraries(ninfer_runtime_support PUBLIC ninfer_core PRIVATE ninfer::json)

# A host outside this build (a C++ bridge from another language) links the public Engine as one
# shared object. The CUDA archives it absorbs are built without position-independent code by
# default, so the option refuses to configure without PIC rather than failing late in a kernel
# archive.
option(NINFER_ENGINE_SHARED "Build the public Engine as a shared object (requires PIC across the tree)" OFF)
if(NINFER_ENGINE_SHARED AND NOT CMAKE_POSITION_INDEPENDENT_CODE)
  message(FATAL_ERROR
    "NINFER_ENGINE_SHARED requires CMAKE_POSITION_INDEPENDENT_CODE=ON: the CUDA archives the "
    "Engine absorbs cannot be relocated into a shared object otherwise.")
endif()
if(NINFER_ENGINE_SHARED)
  set(ninfer_engine_library_type SHARED)
else()
  set(ninfer_engine_library_type STATIC)
endif()

# Public Engine and common contracts; model execution remains in the model layer.
add_library(ninfer_engine ${ninfer_engine_library_type}
  engine/model_instance.cpp
  contract/sampling.cpp
  engine/engine.cpp
)
ninfer_internal_includes(ninfer_engine)
target_include_directories(ninfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ninfer_engine PRIVATE
  ninfer_model_runtime
  ninfer_artifact
  ninfer_core
  ninfer_runtime_support
  ninfer_ops
  ninfer_text
  ninfer_media_decode)
add_library(ninfer::engine ALIAS ninfer_engine)
