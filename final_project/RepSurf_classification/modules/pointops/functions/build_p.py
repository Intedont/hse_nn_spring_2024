import warnings
import os
from torch.utils.cpp_extension import load

warnings.warn("Unable to load pointops_cuda cpp extension.")
pointops_cuda_src = os.path.join(os.path.dirname(__file__), "../src")
pointops_cuda = load('pointops_cuda', [
        pointops_cuda_src + '/pointops_api.cpp',
        pointops_cuda_src + '/ballquery/ballquery_cuda.cpp',
        pointops_cuda_src + '/ballquery/ballquery_cuda_kernel.cu',
        pointops_cuda_src + '/knnquery/knnquery_cuda.cpp',
        pointops_cuda_src + '/knnquery/knnquery_cuda_kernel.cu',
        pointops_cuda_src + '/knnquery_heap/knnquery_heap_cuda.cpp',
        pointops_cuda_src + '/knnquery_heap/knnquery_heap_cuda_kernel.cu',
        pointops_cuda_src + '/grouping/grouping_cuda.cpp',
        pointops_cuda_src + '/grouping/grouping_cuda_kernel.cu',
        pointops_cuda_src + '/grouping_int/grouping_int_cuda.cpp',
        pointops_cuda_src + '/grouping_int/grouping_int_cuda_kernel.cu',
        pointops_cuda_src + '/interpolation/interpolation_cuda.cpp',
        pointops_cuda_src + '/interpolation/interpolation_cuda_kernel.cu',
        pointops_cuda_src + '/sampling/sampling_cuda.cpp',
        pointops_cuda_src + '/sampling/sampling_cuda_kernel.cu'
    ], build_directory=pointops_cuda_src, verbose=False)