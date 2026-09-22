# A foreign host wants one file to link; this project's CUDA archives are built without
# position-independent code, so a shared facade is opt-in rather than the default: turning it on
# without PIC produces a link failure deep in the kernel archives, which is a bad way to learn it.
option(NINFER_CAPI_SHARED "Build the C facade as a shared object (requires PIC across the tree)" OFF)

if(NINFER_CAPI_SHARED AND NOT CMAKE_POSITION_INDEPENDENT_CODE)
  message(FATAL_ERROR
    "NINFER_CAPI_SHARED requires CMAKE_POSITION_INDEPENDENT_CODE=ON: the CUDA archives this "
    "facade absorbs cannot be relocated into a shared object otherwise.")
endif()

if(NINFER_CAPI_SHARED)
  add_library(ninfer_capi SHARED capi.cpp)
  # Only the four entry points leave the object; the engine's own symbols stay inside it.
  set_target_properties(ninfer_capi PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
else()
  add_library(ninfer_capi STATIC capi.cpp)
endif()
ninfer_internal_includes(ninfer_capi)
target_include_directories(ninfer_capi PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ninfer_capi PUBLIC ninfer_engine)
