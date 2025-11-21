#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>



#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"


// This stores the global constants
/*
    存储全局常量：
    scnenName: 场景名称
    numberOfCircles: 圆的数量
    position: 位置； velocity: 速度
    color: 颜色； radius: 半径
    imageWidth: 图像宽度； imageHeight: 图像高度
    imageData: 图像数据
*/
struct GlobalConstants {

    SceneName sceneName;

    int numberOfCircles;

    float* position; 
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
/*
    全局变量，在所有cuda内核中都是只读的。__constant__修饰符表示该变量将存储在GPU上的特殊“常量”内存中。
    （我们在课堂上没有讨论过这种类型的内存，但常量内存是放置只读变量的快速位置）。
*/
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
/* 
    只读查找表，用于快速计算噪声（advanceAnimation需要用于雪花场景）
*/
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
/* 
    颜色渐变表需要用于颜色渐变查找着色器
*/
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// Include parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
/*
    从外部文件包含CUDA代码的部分，以使该文件更简单，并分离不应修改的代码
*/
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


__device__ __inline__ void
shadePixel(float2 pixelCenter, float3 p, float4* imagePtr, int circleIndex) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // Circle does not contribute to the image
    // 如果圆不对图像做出贡献
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // There is a non-zero contribution.  Now compute the shading value

    // Suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks, etc., to implement the conditional.  It
    // would be wise to perform this logic outside of the loops in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    /*
        这是内循环中的条件。虽然它对所有线程的评估结果相同，但在设置通道掩码等以实现条件时会有开销。
        明智的做法是在kernelRenderCircles中的循环之外执行此逻辑。（如果觉得自己不错，可以使用一些专门的模板魔法）。
    */
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z); 
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // Simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // Global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

__global__ void kernelRenderTiling(){
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    if(pos_x > imageWidth || pos_y > imageHeight) return;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    for(int i = 0 ; i < cuConstRendererParams.numberOfCircles ; ++ i){
        float3 p = *(float3*)(&cuConstRendererParams.position[i*3]);
        float rad = cuConstRendererParams.radius[i];

        short minX = static_cast<short>(imageWidth * (p.x - rad));
        short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
        short minY = static_cast<short>(imageHeight * (p.y - rad));
        short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

        if(pos_x >= minX && pos_x <= maxX && pos_y >= minY && pos_y <= maxY){
            float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pos_y * imageWidth + pos_x)]);
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pos_x) + 0.5f),
                                                 invHeight * (static_cast<float>(pos_y) + 0.5f));
            shadePixel(pixelCenterNorm, p, imgPtr, i);
        }
    }

}

__global__ void kernelAdvanceSnowflakes(){
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
    const float dt = 1.f/60.f;
    const float kGravity = -1.8f;
    const float kDragCoeff = 2.f;

    
}

__global__ void kernelAdvanceBouncingBalls(){

}

__global__ void kernelAdvanceHypnosis(){

}

__global__ void kernelAdvanceFireWorks(){
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    
}



void CudaRenderer::advanceAnimation(){
    dim3 blockDim(16,16);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y
    );
    if(sceneName == SNOWFLAKES){
        kernelAdvanceSnowflakes<<<gridDim,blockDim>>>();
    }else if(sceneName == BOUNCING_BALLS){
        kernelAdvanceBouncingBalls<<<gridDim,blockDim>>>();
    }else if(sceneName == HYPNOSIS){
        kernelAdvanceHypnosis<<<gridDim,blockDim>>>();
    }else if(sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim,blockDim>>>();
    }
    cudaDeviceSynchronize();
}

void CudaRenderer::render(){
    dim3 blockDim(16,16);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y
    );
    kernelRenderTiling<<<gridDim,blockDim>>>();
    cudaDeviceSynchronize();
}