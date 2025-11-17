// Define vector unit width here
#define VECTOR_WIDTH 16

#ifndef CMU418INTRIN_H_
#define CMU418INTRIN_H_

#include <cstdlib>
#include <cmath>
#include "logger.h"

//*******************
//* Type Definition *
//*******************

extern Logger CMU418Logger;

template <typename T>
struct __cmu418_vec {
  T value[VECTOR_WIDTH];
};

// Declare a mask with __cmu418_mask
// 中文： 使用cmu418_mask声明一个掩码
struct __cmu418_mask : __cmu418_vec<bool> {};

// Declare a floating point vector register with __cmu418_vec_float
// 中文： 使用cmu418_vec_float声明一个浮点向量寄存器
#define __cmu418_vec_float __cmu418_vec<float>

// Declare an integer vector register with __cmu418_vec_int
// 中文： 使用cmu418_vec_int声明一个整数向量寄存器
#define __cmu418_vec_int   __cmu418_vec<int>

//***********************
//* Function Definition *
//***********************

// Return a mask initialized to 1 in the first N lanes and 0 in the others
// 返回一个掩码，其前N个通道初始化为1，其他通道初始化为0
__cmu418_mask _cmu418_init_ones(int first = VECTOR_WIDTH);

// Return the inverse of maska
// 返回maska的反转
__cmu418_mask _cmu418_mask_not(__cmu418_mask &maska);

// Return (maska | maskb)
__cmu418_mask _cmu418_mask_or(__cmu418_mask &maska, __cmu418_mask &maskb);

// Return (maska & maskb)
__cmu418_mask _cmu418_mask_and(__cmu418_mask &maska, __cmu418_mask &maskb);

// Count the number of 1s in maska
// 返回maska中1的数量
int _cmu418_cntbits(__cmu418_mask &maska);

// Set register to value if vector lane is active
//  otherwise keep the old value
// 设置寄存器的值（如果向量通道处于活动状态），否则保持旧值
void _cmu418_vset_float(__cmu418_vec_float &vecResult, float value, __cmu418_mask &mask);
void _cmu418_vset_int(__cmu418_vec_int &vecResult, int value, __cmu418_mask &mask);
// For user's convenience, returns a vector register with all lanes initialized to value
// 方便用户使用，返回一个所有通道都初始化为value的向量寄存器
__cmu418_vec_float _cmu418_vset_float(float value);
__cmu418_vec_int _cmu418_vset_int(int value);

// Copy values from vector register src to vector register dest if vector lane active
// otherwise keep the old value
// 从向量寄存器src复制值到向量寄存器dest（如果向量通道处于活动状态），否则保持旧值
void _cmu418_vmove_float(__cmu418_vec_float &dest, __cmu418_vec_float &src, __cmu418_mask &mask);
void _cmu418_vmove_int(__cmu418_vec_int &dest, __cmu418_vec_int &src, __cmu418_mask &mask);

// Load values from array src to vector register dest if vector lane active
//  otherwise keep the old value
// 从数组src加载值到向量寄存器dest（如果向量通道处于活动状态），否则保持旧值
void _cmu418_vload_float(__cmu418_vec_float &dest, float* src, __cmu418_mask &mask);
void _cmu418_vload_int(__cmu418_vec_int &dest, int* src, __cmu418_mask &mask);

// Store values from vector register src to array dest if vector lane active
//  otherwise keep the old value
// 从向量寄存器src存储值到数组dest（如果向量通道处于活动状态），否则保持旧值
void _cmu418_vstore_float(float* dest, __cmu418_vec_float &src, __cmu418_mask &mask);
void _cmu418_vstore_int(int* dest, __cmu418_vec_int &src, __cmu418_mask &mask);

// Return calculation of (veca + vecb) if vector lane active
//  otherwise keep the old value
// 返回（veca + vecb）的计算结果（如果向量通道处于活动状态），否则保持旧值
void _cmu418_vadd_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_vadd_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// Return calculation of (veca - vecb) if vector lane active
//  otherwise keep the old value
// 返回（veca - vecb）的计算结果（如果向量通道处于活动状态），否则保持旧值
void _cmu418_vsub_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_vsub_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// Return calculation of (veca * vecb) if vector lane active
//  otherwise keep the old value
void _cmu418_vmult_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_vmult_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// Return calculation of (veca / vecb) if vector lane active
//  otherwise keep the old value
void _cmu418_vdiv_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_vdiv_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// Return calculation of (veca >> vecb) if vector lane active
//  otherwise keep the old value
void _cmu418_vshiftright_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// Return calculation of (veca & vecb) if vector lane active
//  otherwise keep the old value
void _cmu418_vbitand_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// Return calculation of absolute value abs(veca) if vector lane active
//  otherwise keep the old value
void _cmu418_vabs_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &veca, __cmu418_mask &mask);
void _cmu418_vabs_int(__cmu418_vec_int &vecResult, __cmu418_vec_int &veca, __cmu418_mask &mask);

// Return a mask of (veca > vecb) if vector lane active
//  otherwise keep the old value
void _cmu418_vgt_float(__cmu418_mask &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_vgt_int(__cmu418_mask &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// Return a mask of (veca < vecb) if vector lane active
//  otherwise keep the old value
void _cmu418_vlt_float(__cmu418_mask &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_vlt_int(__cmu418_mask &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// Return a mask of (veca == vecb) if vector lane active
//  otherwise keep the old value
void _cmu418_veq_float(__cmu418_mask &vecResult, __cmu418_vec_float &veca, __cmu418_vec_float &vecb, __cmu418_mask &mask);
void _cmu418_veq_int(__cmu418_mask &vecResult, __cmu418_vec_int &veca, __cmu418_vec_int &vecb, __cmu418_mask &mask);

// Adds up adjacent pairs of elements, so
//  [0 1 2 3] -> [0+1 0+1 2+3 2+3]
// 中文：相邻元素两两相加，
void _cmu418_hadd_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &vec);

// Performs an even-odd interleaving where all even-indexed elements move to front half
//  of the array and odd-indexed to the back half, so
//  [0 1 2 3 4 5 6 7] -> [0 2 4 6 1 3 5 7]
// 中文：执行偶数-奇数交错，所有偶数索引元素移动到数组的前半部分，奇数索引元素移动到后半部分，
void _cmu418_interleave_float(__cmu418_vec_float &vecResult, __cmu418_vec_float &vec);

// Add a customized log to help debugging
// 为了帮助调试，添加一个自定义日志
void addUserLog(const char * logStr);

#endif
