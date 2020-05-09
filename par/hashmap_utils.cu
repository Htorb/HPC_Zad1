#include "hashmap_utils.h"

__host__  __device__ int h1(int Ci) {
    return 7 * Ci + 5;
}

__host__  __device__ int h2(int Ci) {
    return 2 * Ci + 1; 
}
//size > 0 !!
__host__  __device__ int double_hash(int Ci, int itr, int size) { 
    return (h1(Ci) + itr * h2(Ci)) % size; //maybe can be improved
}

__device__ void hashmap_insert(int* hashComm, float* hashWeight, int offset, int size, int cj, float w) {
    int pos, itr = 0;
    do {
        pos = offset + double_hash(cj, itr, size);

        if (hashComm[pos] == cj) {
                atomicAdd(&hashWeight[pos], w); 
        } else if (hashComm[pos] == EMPTY_SLOT) {
            if (cj == atomicCAS(&hashComm[pos], EMPTY_SLOT, cj)) {
                    atomicAdd(&hashWeight[pos], w); 
            } 
            else if (hashComm[pos] == cj) {
                    atomicAdd(&hashWeight[pos], w); 
            }
        }
        itr++;
    } while (hashComm[pos] != cj);
}

void hashmap_create(dvi& hashSize, dvi& hashOffset, dvi& hashComm, dvf& hashWeight) {        
    hashOffset = dvi(hashSize.size() + 1); 
    hashOffset[0] = 0;
    thrust_inclusive_scan_with_shift(hashSize, hashOffset, 1);
    hashComm = dvi(hashOffset.back(), EMPTY_SLOT);
    hashWeight = dvf(hashOffset.back(), 0);
}

__device__ int hashmap_find(int* hashComm, int offset, int size, int ci) {
    int pos, itr = 0;
    do {    
        pos = offset + double_hash(ci, itr, size);
        itr++;
    } while (hashComm[pos] != ci && hashComm[pos] != EMPTY_SLOT);
    return pos;
}
