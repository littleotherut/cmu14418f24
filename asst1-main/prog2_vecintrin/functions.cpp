#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "CMU418intrin.h"
#include "logger.h"
using namespace std;


void absSerial(float* values, float* output, int N) {
    for (int i=0; i<N; i++) {
		float x = values[i];
		if (x < 0) {
			output[i] = -x;
		} else {
			output[i] = x;
		}
    }
}

// implementation of absolute value using 15418 instrinsics
void absVector(float* values, float* output, int N) {
    __cmu418_vec_float x;
    __cmu418_vec_float result;
    __cmu418_vec_float zero = _cmu418_vset_float(0.f);
    __cmu418_mask maskAll, maskIsNegative, maskIsNotNegative;

    //  Note: Take a careful look at this loop indexing.  This example
    //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
    //  Why is that the case?
    for (int i=0; i<N; i+=VECTOR_WIDTH) {

	// All ones
	maskAll = _cmu418_init_ones();

	// All zeros
	maskIsNegative = _cmu418_init_ones(0);

	// Load vector of values from contiguous memory addresses
	_cmu418_vload_float(x, values+i, maskAll);               // x = values[i];

	// Set mask according to predicate
	_cmu418_vlt_float(maskIsNegative, x, zero, maskAll);     // if (x < 0) {

	// Execute instruction using mask ("if" clause)
	_cmu418_vsub_float(result, zero, x, maskIsNegative);      //   output[i] = -x;

	// Inverse maskIsNegative to generate "else" mask
	maskIsNotNegative = _cmu418_mask_not(maskIsNegative);     // } else {

	// Execute instruction ("else" clause)
	_cmu418_vload_float(result, values+i, maskIsNotNegative); //   output[i] = x; }

	// Write results back to memory
	_cmu418_vstore_float(output+i, result, maskAll);
    }
}

// Accepts an array of values and an array of exponents
// For each element, compute values[i]^exponents[i] and clamp value to
// 4.18.  Store result in outputs.
// Uses iterative squaring, so that total iterations is proportional
// to the log_2 of the exponent
/* 中文：接收一个值数组和一个指数数组
 对于每个元素，计算values[i]^exponents[i]并将值夹紧到4.18。
 将结果存储在输出中。
 使用迭代平方，因此总迭代次数与指数的log_2成正比 
*/
void clampedExpSerial(float* values, int* exponents, float* output, int N) {
    for (int i=0; i<N; i++) {
		float x = values[i];
		float result = 1.f;
		int y = exponents[i];
		float xpower = x;
		while (y > 0) {
			if (y & 0x1) {
				result *= xpower;
			}
			xpower = xpower * xpower;
			y >>= 1;
		}
		if (result > 4.18f) {
			result = 4.18f;
		}
		output[i] = result;
    }
}

void clampedExpVector(float* values, int* exponents, float* output, int N) {
    // Implement your vectorized version of clampedExpSerial here
    //  ...
	__cmu418_vec_int vecZero = _cmu418_vset_int(0),vecOne = _cmu418_vset_int(1);
	__cmu418_vec_float vec418 = _cmu418_vset_float(4.18f);
	for(int i = 0 ; i < N ; i += VECTOR_WIDTH){
		int width = min(VECTOR_WIDTH,N - i);
		__cmu418_mask vecMaskOne = _cmu418_init_ones(width);
		
		__cmu418_vec_float x;
		_cmu418_vload_float(x,values+i,vecMaskOne);
		__cmu418_vec_int y;
		_cmu418_vload_int(y,exponents+i,vecMaskOne);

		__cmu418_vec_float vecResult = _cmu418_vset_float(1.f);
		__cmu418_mask activate = _cmu418_init_ones(0);
		__cmu418_mask v418 = _cmu418_mask_not(vecMaskOne);
		
		_cmu418_vgt_int(activate,y,vecZero,vecMaskOne);
		while(_cmu418_cntbits(activate)){
			__cmu418_vec_int vba ;
			__cmu418_mask vbaMask = vbaMask = _cmu418_init_ones(0);
			_cmu418_vbitand_int(vba,y,vecOne,activate);
			_cmu418_veq_int(vbaMask,vba,vecOne,activate);
			_cmu418_vmult_float(vecResult,vecResult,x,vbaMask);
			_cmu418_vmult_float(x,x,x,activate);
			_cmu418_vshiftright_int(y,y,vecOne,activate);
			_cmu418_vgt_int(activate,y,vecZero,vecMaskOne);	
		}
		_cmu418_vgt_float(v418,vecResult,vec418,vecMaskOne);
		_cmu418_vset_float(vecResult,4.18f,v418);
		_cmu418_vstore_float(output+i,vecResult,vecMaskOne);
	}
}


float arraySumSerial(float* values, int N) {
    float sum = 0;
    for (int i=0; i<N; i++) {
		sum += values[i];
    }

    return sum;
}

// Assume N % VECTOR_WIDTH == 0
// Assume VECTOR_WIDTH is a power of 2
// 假设N % VECTOR_WIDTH == 0
// 假设VECTOR_WIDTH是2的幂
float arraySumVector(float* values, int N) {
    // Implement your vectorized version here
    //  ...
	__cmu418_vec_float vsum = _cmu418_vset_float(0.f);
	for(int i = 0 ; i < N ; i += VECTOR_WIDTH){
		__cmu418_mask mask = _cmu418_init_ones(min(VECTOR_WIDTH,N - i));
		__cmu418_vec_float x;
		_cmu418_vload_float(x,values+i,mask);
		_cmu418_vadd_float(vsum,vsum,x,mask);
	}
	for(int i = VECTOR_WIDTH ; i > 1 ; i >>=1){
		_cmu418_hadd_float(vsum,vsum);
		_cmu418_interleave_float(vsum,vsum);
		// for(int j = 0 ; j < i ; ++ j){
		// 	printf("%f ",vsum.value[j]);
		// }
		// printf("\n");
	}
	float sum = 0;
	__cmu418_mask mask = _cmu418_init_ones(1);
	_cmu418_vstore_float(&sum,vsum,mask);
	return sum;
}
