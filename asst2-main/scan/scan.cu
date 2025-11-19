#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"


extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2.
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void prescan_block(int *output_global, int *input_global,int *block_sums, const int n){
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int block_size = blockDim.x*2;
    int blockOffset = blockIdx.x * block_size;
    int offset = 1;

    int idx1 = blockOffset + 2*tid, idx2 = blockOffset + 2*tid + 1;
    temp[tid*2] = (idx1 < n) ? input_global[idx1] : 0;
    temp[tid*2+1] = (idx2 < n) ? input_global[idx2] : 0;

    for(int d = block_size >> 1 ; d > 0 ; d >>= 1){
        __syncthreads();
        if(tid < d){
            int idx1 = offset * (2*tid+1) - 1;
            int idx2 = offset * (2*tid+2) - 1;
            temp[idx2] += temp[idx1];
        }
        offset *= 2;
    }
    if(tid == 0){
        if( block_sums != NULL )
            block_sums[blockIdx.x] = temp[block_size-1];
        temp[block_size-1]=0;
    }
    for(int i = 1 ; i < block_size ; i *= 2){
        offset>>=1;
        __syncthreads();
        if(tid < i){
            int idx1 = offset*(2*tid+1)-1;
            int idx2 = offset*(2*tid+2)-1;
            int t = temp[idx1];
            temp[idx1] = temp[idx2];
            temp[idx2] += t;
        }
    }
    __syncthreads();
    if(idx1 < n) output_global[idx1] = temp[2*tid];
    if(idx2 < n) output_global[idx2] = temp[2*tid+1];
}

__global__ void add_block_sums(int *output_global, int *block_sums, const int n){
    const int block_offset = blockIdx.x * (blockDim.x * 2);
    if(blockIdx.x == 0) return;

    int add_val = block_sums[blockIdx.x]; // 每个块加上它之前的块的和
    int idx1 = block_offset + 2*threadIdx.x, idx2 = block_offset + 2*threadIdx.x + 1;
    if(idx1 < n) output_global[idx1] += add_val;
    if(idx2 < n) output_global[idx2] += add_val;
}

void exclusive_scan(int* device_data, int length)
{
    /* TODO
     * Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the data in device memory
     * The data are initialized to the inputs.  Your code should
     * do an in-place scan, generating the results in the same array.
     * This is host code -- you will need to declare one or more CUDA
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the data array is sized to accommodate the next
     * power of 2 larger than the input.
     */
    /*
        完成这个函数，使用你的exclusive scan实现。
        传递给你的数据位于设备内存中。
        数据已初始化为输入。你的代码应该进行就地扫描，在同一个数组中生成结果。
        这是主机代码——你需要声明一个或多个CUDA内核（使用__global__装饰器）以便在GPU上并行运行代码。
        注意你得到了数组的实际长度，但可以假设数据数组的大小足以容纳大于输入的下一个的2的幂。
    */
    length = nextPow2(length);
    dim3 blockDim = (256);
    dim3 gridDim = ((length + blockDim.x * 2 - 1) / (blockDim.x * 2));
    int *device_block_sums = nullptr;
    if (gridDim.x > 1)
        cudaMalloc(&device_block_sums, nextPow2(gridDim.x) * sizeof(int));
    prescan_block<<<gridDim, blockDim, sizeof(int) * blockDim.x * 2>>>(device_data, device_data,
                                                    device_block_sums, length);
    if (gridDim.x > 1) {
        exclusive_scan(device_block_sums, gridDim.x); // 递归处理区块和
        add_block_sums<<<gridDim.x, blockDim>>>(device_data, device_block_sums, length); // 加上区块和
    }
    cudaFree(device_block_sums);
}
/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
/*
这个函数是你将编写的代码的包装器——它将输入复制到GPU并计时对上面的exclusive_scan()函数的调用。
你不应该修改它。
*/
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_data;
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness.
    // You may have an easier time in your implementation if you assume the
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);
    // printf("Input: ");
    // for(int i = 0 ; i < 10 ; i++){
    //     printf("%d ",inarray[i]);
    // }
    // printf("\n");
    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_data, end - inarray);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    // printf("Output: ");
    // for(int i = 0 ; i < 10 ; i++){
    //     printf("%d ",resultarray[i]);
    // }
    // printf("\n");
    // printf("%d\n",end-inarray);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
/*
    Thrust库的独占扫描函数的包装器
    如上所述，将输入复制到GPU上，并仅计时扫描本身的执行。
    你不需要产生与Thrust版本竞争的性能。
*/
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void peak_find(int *data, int *result, const int n){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx < n-1 && idx > 0){
        result[idx] = (data[idx]>data[idx-1]&&data[idx]>data[idx+1]) ?
            1:0;
    }
}
__global__ void result(int *data,int *result, const int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n-1 && idx > 0 && data[idx]<data[idx+1]){
        result[data[idx]]=idx;
    }
}

int find_peaks(int *device_input, int length, int *device_output) {
    /* TODO:
     * Finds all elements in the list that are greater than the elements before and after,
     * storing the index of the element into device_result.
     * Returns the number of peak elements found.
     * By definition, neither element 0 nor element length-1 is a peak.
     *
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if
     * it requires that. However, you must ensure that the results of
     * find_peaks are correct given the original length.
     */
    /*
        找出列表中大于前后元素的所有元素，并将该元素的索引存储到device_result中。
        返回找到的峰值元素的数量。
        根据定义，元素0和元素length-1都不是峰值。
        你的任务是实现这个函数。你可能需要使用对exclusive_scan()的一个或多个调用，以及额外的CUDA内核启动。
        注意：与扫描代码一样，我们确保分配的数组大小为2的幂，因此如果需要，你可以使用你的exclusive_scan函数。
        但是，你必须确保find_peaks的结果是正确的，给定原始长度。
    */
    int roundlength = nextPow2(length), ans;
    dim3 blockDim = (256);
    dim3 gridDim = (length + blockDim.x - 1)/ blockDim.x;
    int *tmp;
    cudaMalloc(&tmp,roundlength*sizeof(int));
    peak_find<<<gridDim,blockDim>>>(device_input,tmp,length);
    exclusive_scan(tmp,roundlength);
    result<<<gridDim,blockDim>>>(tmp,device_output,length);
    cudaMemcpy(&ans,tmp+roundlength-1,sizeof(int),cudaMemcpyDeviceToHost);
    return ans;
}



/* Timing wrapper around find_peaks. You should not modify this function.
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int result = find_peaks(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}


void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
