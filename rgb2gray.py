import numpy as np
import pycuda.autoinit

from pycuda import compiler, gpuarray

kernel_code_template = """
    __global__ void rgb2gray(unsigned int *grayImage, unsigned int *rgbImage)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if(x < %(width)s && y < %(height)s) {

            int grayOffset = y * %(width)s + x;
            int rgbOffset = grayOffset * %(channels)s;

            unsigned int r = rgbImage[rgbOffset];
            unsigned int g = rgbImage[rgbOffset + 1]; 
            unsigned int b = rgbImage[rgbOffset + 2];

            grayImage[grayOffset] = int((r + g + b) / 3); 
        }
    }
"""

def rgb2gray(image, height, width, channels=3):
    a_cpu = np.array(image).astype(np.uint8)
    b_cpu = np.zeros((height, width)).astype(np.uint8)

    a_gpu = gpuarray.to_gpu(a_cpu) 
    b_gpu = gpuarray.to_gpu(b_cpu)

    kernel_code = kernel_code_template % {
        'width': str(width),
        'height': str(height),
        'channels': str(channels)
    }

    mod = compiler.SourceModule(kernel_code)
    matrixmul = mod.get_function('rgb2gray')
    matrixmul(
        b_gpu,
        a_gpu, 
        block=(6,36, 1),
        grid = (100,8,1)
    )

    return b_gpu.get()