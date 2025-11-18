#include <stdio.h>
#include <pthread.h>

// Use this code to time your threads
#include "CycleTimer.h"


/*

  15418 Spring 2012 note: This code was modified from example code
  originally provided by Intel.  To comply with Intel's open source
  licensing agreement, their copyright is retained below.

  -----------------------------------------------------------------

  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


// Core computation of Mandelbrot set membershop
// Iterate complex number c to determine whether it diverges
// 中文： 判断复数c是否发散
static inline int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

//
// MandelbrotSerial --
//
// Compute an image visualizing the mandelbrot set.  The resulting
// array contains the number of iterations required before the complex
// number corresponding to a pixel could be rejected from the set.
//
// * x0, y0, x1, y1 describe the complex coordinates mapping
//   into the image viewport.
// * width, height describe the size of the output image
// * startRow, endRow describe how much of the image to compute
/*
中文： 计算曼德尔布罗特集图像。结果数组包含每个像素对应的复数在被拒绝出集合之前所需的迭代次数。
* x0, y0, x1, y1 描述映射到图像视口的复数坐标。
* width, height 描述输出图像的大小
* startRow, endRow 描述要计算的图像部分
*/
void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int endRow,
    int maxIterations,
    int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    for (int j = startRow; j < endRow; j++) {
        for (int i = 0; i < width; ++i) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = (j * width + i);
            output[index] = mandel(x, y, maxIterations);
        }
    }
}


// Struct for passing arguments to thread routine
// 中文： 用于传递线程例程参数的结构体
typedef struct {
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations; // 最大迭代次数
    int* output; // 输出数组
    int threadId; 
    int numThreads; // 线程总数
    int startRow; // 起始行
    int endRow;   // 结束行
} WorkerArgs;



//
// workerThreadStart --
//
// Thread entrypoint.
// 中文： 线程入口点。
void* workerThreadStart(void* threadArgs) {
    // 记录每个线程用时
    double startTime = CycleTimer::currentSeconds();
    WorkerArgs* args = static_cast<WorkerArgs*>(threadArgs);

    mandelbrotSerial(
        args->x0,args->y0,args->x1,args->y1,
        args->width,args->height,
        args->startRow,args->endRow,
        args->maxIterations,
        args->output
    );
    double endTime = CycleTimer::currentSeconds();
    // printf("Thread %d: Time taken = %.4f sec\n", args->threadId, endTime - startTime);
    // printf("Hello world from thread %d\n", args->threadId);

    return NULL;
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Multi-threading performed via pthreads.
// 中文： 曼德尔布罗特集图像生成的多线程实现。多线程通过 pthreads 实现。
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    const static int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    pthread_t workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    int weight = numThreads > 1 ? (1+numThreads/2)*numThreads/2 : 1;
    int perThreadHeight = height / weight;
    args[0].startRow = 0;
    args[0].endRow = perThreadHeight * (numThreads/2);
    args[numThreads - 1].endRow = height;
    args[numThreads - 1].startRow = height - perThreadHeight * (numThreads/2);
    for(int i = 1 ; i < numThreads/2 ; ++ i){
        args[i].startRow = args[i-1].endRow; 
        args[i].endRow = args[i].startRow + perThreadHeight * (numThreads/2 - i);
        args[numThreads - i - 1].endRow = args[numThreads - i].startRow;
        args[numThreads - i - 1].startRow = args[numThreads - i - 1].endRow - perThreadHeight * (numThreads/2 - i);
        // printf("Thread %d: startRow=%d, endRow=%d\n", i, args[i].startRow, args[i].endRow);
        // printf("Thread %d: startRow=%d, endRow=%d\n", numThreads - i - 1, args[numThreads - i - 1].startRow, args[numThreads - i - 1].endRow);
    }
    args[numThreads/2-1].endRow = height/2;
    args[numThreads/2].startRow = height/2;
    // printf("perThreadHeight=%d,nums=%d,height=%d\n", perThreadHeight, numThreads, height);
    for (int i=0; i<numThreads; i++) {
        // TODO: Set thread arguments here.
        args[i].threadId = i;
        args[i].x0 = x0;args[i].x1 = x1;
        args[i].y0 = y0;
        args[i].y1 = y1;
        args[i].numThreads = numThreads;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].output = output;
        // printf("Thread %d: height=%d, width=%d\n", i,args[i].height, args[i].width);
    }


    // Fire up the worker threads.  Note that numThreads-1 pthreads
    // are created and the main app thread is used as a worker as
    // well.
    // 中文： 启动工作线程。注意，创建了 numThreads-1 个 pthreads，主应用程序线程也用作工作线程。
    for (int i=1; i<numThreads; i++)
        pthread_create(&workers[i], NULL, workerThreadStart, &args[i]);

    workerThreadStart(&args[0]);

    // wait for worker threads to complete
    for (int i=1; i<numThreads; i++)
        pthread_join(workers[i], NULL);
}
