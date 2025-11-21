#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"


class CudaRenderer : public CircleRenderer {

private:

    Image* image;
    SceneName sceneName;

    int numberOfCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;

    // Binning buffers
    int* cudaDeviceTileCounts;       // 每个 tile 包含的圆数量
    int* cudaDeviceTileOffsets;      // 前缀和，每个 tile 在 list 中的起始位置
    int* cudaDeviceTileCircleList;   // 所有 tile 的圆索引列表
    int* cudaDeviceTileWritePtr;     // 用于原子写入的临时指针
    
    int tilesX;
    int tilesY;
    int totalTiles;
    int maxCirclesPerTile;           // 保守估计的上界


public:

    CudaRenderer();
    virtual ~CudaRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();

    void shadePixel(
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData, 
        int circleIndex);

    void buildTileBins();
};


#endif
