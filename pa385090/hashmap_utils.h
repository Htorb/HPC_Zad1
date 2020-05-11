#ifndef HASHMAP_UTILS_H
#define HASHMAP_UTILS_H

#include "thrust_wrappers.h"

#define EMPTY_SLOT -1

__host__  __device__ int h1(int);
__host__  __device__ int h2(int);
__host__  __device__ int double_hash(int, int, int);

__device__ void hashmap_insert(int*, float*, int, int, int, float);
void hashmap_create(dvi&, dvi&, dvi&, dvf&);
__device__ int hashmap_find(int*, int, int, int);


#endif