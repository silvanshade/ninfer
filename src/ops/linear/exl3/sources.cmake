target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/exl3_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../../wrapper/separate_projection.cpp"
)

# The vendored exllamav3 kernels define device globals per translation unit and launch
# cooperatively; like the NVFP4 kernels they build as a self-contained non-RDC archive.
target_sources(ninfer_exl3_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/exl3_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_reconstruct.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_unit_b1.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_unit_b2.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_unit_b3.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_unit_b4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_unit_b5.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_unit_b6.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_unit_b7.cu"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_unit_b8.cu"
)
