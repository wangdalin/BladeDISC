resolve_env(TORCH_BLADE_BUILD_TENSORRT OFF)

if (TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT)
  if (TORCH_BLADE_BUILD_TENSORRT)
    include(cmake/tensorrt.cmake)
  endif (TORCH_BLADE_BUILD_TENSORRT)
endif (TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT)
