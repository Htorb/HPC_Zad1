#include <iostream>
#include <cstring>
#include <vector>
#include <map>
#include <set>
#include <cassert> 
#include <string>

#include "thrust_wrappers.h"
#include "utils.h"
#include "hashmap_utils.h"


//ASSUMPTIONS 
// there is at least one edge with a positive weight

// TODO bucket partition
// TODO how can update modularity
// TODO code grooming
// TODO correct outer loop condition (from slack)


#define ptr(a) \
    thrust::raw_pointer_cast((a).data())

#define NO_EDGE 0
#define BLOCKS_NUMBER 16
#define THREADS_PER_BLOCK 128

using namespace std;

bool DEBUG = false;


__device__ void updateMaxModularity(int* maxC, float* maxDeltaMod, int newC, float newDeltaMod) {
    if (newDeltaMod > *maxDeltaMod || newDeltaMod == *maxDeltaMod && newC < *maxC) {
        *maxC = newC;
        *maxDeltaMod = newDeltaMod;
    }
}


__global__ void computeMoveGPU(int n,
                                int* newComm, 
                                int* V,
                                int* N, 
                                float* W, 
                                int* C,
                                int* comSize, 
                                float* k, 
                                float* ac, 
                                const float wm,
                                int* hashOffset,
                                float* hashWeight,
                                int* hashComm) {
    __shared__ int partialCMax[THREADS_PER_BLOCK];
    __shared__ float partialDeltaMod[THREADS_PER_BLOCK];
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        int tid = threadIdx.x;
        int step = blockDim.x;
        int offset = hashOffset[i];
        int size = hashOffset[i + 1] - offset;
        int ci = C[i];
        int pos;

        if (size == 0) {
            continue;
        }

        for (int j = tid + V[i]; j < V[i + 1]; j += step) {
            if (W[j] == NO_EDGE)
                break;
            if (N[j] != i) {
                hashMapInsert(hashComm, hashWeight, offset, size, C[N[j]], W[j]);
            }
        }
        __syncthreads();

        partialCMax[tid] = n;
        partialDeltaMod[tid] = -1;
        for (pos = offset + tid; pos < offset + size; pos += step) {
            if (hashComm[pos] == EMPTY_SLOT)
                continue;

            int newC = hashComm[pos];
            
            float deltaMod = hashWeight[pos] / wm 
                                + k[i] * (ac[ci] - k[i] - ac[newC]) / (2 * wm * wm);
        
            if (comSize[newC] > 1 || comSize[ci] > 1 || newC < ci) {
                updateMaxModularity(&partialCMax[tid], &partialDeltaMod[tid], newC, deltaMod);
            }
        }
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0 ; s >>= 1) {
            if (tid < s) {
                updateMaxModularity(&partialCMax[tid], &partialDeltaMod[tid], partialCMax[tid + s], partialDeltaMod[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            pos = hashMapFind(hashComm, offset, size, ci);

            if (partialDeltaMod[0] - hashWeight[pos] / wm > 0) {
                newComm[i] = partialCMax[0];
            } else {
                newComm[i] = ci;
            }
        }
    }
}


//WARNING WORKS ONLY WITH ONE KERNEL
__global__ void calculateModularityGPU(int n,
                                        int c,
                                        int* V,
                                        int* N, 
                                        float* W, 
                                        int* C,
                                        int* uniqueC, 
                                        float* ac, 
                                        const float wm,
                                        float* Q) {
    __shared__ float partials[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int step  = blockDim.x;

    float a = 0;
    for (int i = tid; i < n; i += step) {
        for (int j = V[i]; j < V[i + 1]; ++j) {
            if (C[N[j]] == C[i]) {
                a += W[j] / (2 * wm);
            }
        }
    }
    for (int i = tid; i < c; i += step) {
        a -= ac[uniqueC[i]] * ac[uniqueC[i]] / (4 * wm * wm);
    }
    partials[tid] = a;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partials[tid] += partials[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *Q = partials[0];
    }
}

__global__ void initializeKGPU(int n, const int* V, const float* W, float* k) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        for (int j = V[i]; j < V[i + 1]; ++j) {
            if (W[j] == NO_EDGE)
                break;
            k[i] += W[j];
        } 
    }
}


__global__ void initializeAcGPU(int n, int* C, float* k, float* ac) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&ac[C[i]], k[i]);
    }
}


void initializeUniqueCAndCGPU(int n, const dvi& C, dvi& uniqueC, int& c) {
    uniqueC = C;
    thrust_sort(uniqueC);
    thrust_unique(uniqueC);
    c = uniqueC.size();
}
        


__global__ void initializeDegreeGPU(int n, int* V, float* W, int* degree) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int ctr = 0;
        for (int j = V[i]; j < V[i + 1]; ++j) {
            if (W[j] == NO_EDGE)
                break;
            ctr++;
        }
        degree[i] = ctr;
    }
}


__global__ void initializeComSizeGPU(int n, int* C, int* comSize) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&comSize[C[i]], 1);
    }
}



__global__ void initializeComDegreeGPU(int n, int* degree, int* C, int* comDegree) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&comDegree[C[i]], degree[i]); 
    }
}


__global__ void initializeNewIDGPU(int n, int* C, int* comSize, int* newID) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (comSize[C[i]] != 0) {
            atomicCAS(&newID[C[i]], 0, 1);
        }
    }
}

__global__ void initializeCommGPU(int n, int* C, int* comm, int* vertexStart) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int res = atomicSub(&vertexStart[C[i]], 1) - 1;
        comm[res] = i; 
    }
}


__global__ void initializeNewVGPU(int n, int* C, int* newID, int* edgePos, int* newV) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicCAS(&newV[newID[C[i]] + 1], 0, edgePos[C[i]]);
    }
}

__global__ void saveFinalCommunitiesGPU(int initialN,
                                        int* finalC,
                                        int* C,
                                        int* newID) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < initialN; i += blockDim.x * gridDim.x) {
        finalC[i] = newID[C[finalC[i]]];
    }
}



__global__ void mergeCommunityFillHashMapGPU(int n,
                                                int* V,
                                                int* N,
                                                float* W,
                                                int* C,
                                                int* comm,
                                                int* degree,
                                                int* newID,
                                                int* hashOffset,
                                                int* hashComm,
                                                float* hashWeight,
                                                bool DEBUG) {
    for (int idx = blockIdx.x; idx < n; idx += gridDim.x) {
        int tid = threadIdx.x;
        int step = blockDim.x;

        int i = comm[idx];
        int newci = newID[C[i]];
        int offset = hashOffset[newci];
        int size = hashOffset[newci + 1] - offset;

        if (size == 0) {
            continue;
        }

        if (DEBUG) {
            assert(size >= degree[i]);
        }

        for (int j = tid + V[i]; j < V[i + 1]; j += step) {
            if (W[j] == NO_EDGE)
                break; 

            hashMapInsert(hashComm, hashWeight, offset, size, newID[C[N[j]]], W[j]);
        }   
    }
}


__global__ void mergeCommunityInitializeGraphGPU(int* hashOffset,
                                                    int* hashComm,
                                                    float* hashWeight,
                                                    int newn,
                                                    int* newV,
                                                    int* newN,
                                                    float* newW) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < newn; i += blockDim.x * gridDim.x) {
        int edgeId = newV[i];
        for (int pos = hashOffset[i]; pos < hashOffset[i + 1]; ++pos) {
            int newcj = hashComm[pos];

            if (newcj == EMPTY_SLOT) {
                continue;
            }
            float wsum = hashWeight[pos];

            newN[edgeId] = newcj;
            newW[edgeId] = wsum;
            edgeId++;
        }        
    }
}


int main(int argc, char *argv[]) {
    //commandline vars
    bool showAssignment = false;
    float threshold = 0;
    string matrixFile;

    //graph vars host
    int n; //number vertices 
    int m; //number of edges
    dvi V; //vertices
    dvi N; //neighbours
    dvf W; //weights
    float wm; //sum of weights
    dvi C; //current clustering
    dvi newComm; //temporary array to store new communities
    dvf k; //sum of vertex's edges
    dvf ac; //sum of cluster edges
    int c; //number of communities
    dvi uniqueC; //list of unique communities ids
    dvi comSize; //size of ech community
    dvi degree; //degree of each vertex

    

    int initialN; //number of vertices in the first iteration
    dvi finalC; //final clustering result 


    float Qba, Qp, Qc; //modularity before outermostloop iteration, before and after modularity optimisation respectively
    
    cudaEvent_t startTime, stopTime;



    parseCommandline(showAssignment, threshold, matrixFile, argc, argv, DEBUG);

    vi tmpV;
    vi tmpN;
    vf tmpW;
    readGraphFromFile(matrixFile, n, m, tmpV, tmpN, tmpW);
    V = tmpV;
    N = tmpN;
    W = tmpW;

    startRecordingTime(startTime, stopTime);
 
    initialN = n;
    wm = thrust_sum(W) / 2;

    finalC = dvi(n);
    thrust_sequence(finalC); 

    do { 
        C = dvi(n);
        thrust_sequence(C); 

        k = vf(n, 0);
        initializeKGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, 
                                                          ptr(V), 
                                                          ptr(W), 
                                                          ptr(k)); 

        
        if (DEBUG) {
            float ksum = thrust_sum(k);
            assert(abs(ksum - 2 * wm) < 0.0001);
        }

        ac = k; 
        
        //modularity optimisation phase

        initializeUniqueCAndCGPU(n, C, uniqueC, c);


        dvf dQc(1);
        calculateModularityGPU<<<1, THREADS_PER_BLOCK>>>(n,
                                                        c, 
                                                        ptr(V),
                                                        ptr(N),
                                                        ptr(W),
                                                        ptr(C),
                                                        ptr(uniqueC),
                                                        ptr(ac),
                                                        wm,
                                                        ptr(dQc));
        Qc = dQc[0];
        Qba = Qc;

        
        cerr << "modularity: " << Qc << endl;
        do {
            
            newComm = C;
            comSize = dvi(n, 0);
            initializeComSizeGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(C), ptr(comSize));
            degree = dvi(n, 0);
            initializeDegreeGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(V), ptr(W), ptr(degree));
            
            
            dvi hashSize = dvi(n);
            thrust_transform_hashmap_size(degree, hashSize, 1.5);

            dvi hashOffset;
            dvi hashComm;
            dvf hashWeight;
            hashMapCreate(hashSize, hashOffset, hashComm, hashWeight);

          
            computeMoveGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, 
                                                                 ptr(newComm), 
                                                                 ptr(V), 
                                                                 ptr(N), 
                                                                 ptr(W), 
                                                                 ptr(C), 
                                                                 ptr(comSize), 
                                                                 ptr(k), 
                                                                 ptr(ac), 
                                                                 wm, 
                                                                 ptr(hashOffset), 
                                                                 ptr(hashWeight), 
                                                                 ptr(hashComm));
            
            C = newComm;

            ac.assign(n, 0);
            initializeAcGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(C), ptr(k), ptr(ac));
            if (DEBUG) {
                float acsum = thrust_sum(ac);                             
                assert(abs(acsum - 2 * wm) < 0.0001);
            }
            

            Qp = Qc;
            initializeUniqueCAndCGPU(n, C, uniqueC, c);
            dvf dQc(1);
            calculateModularityGPU<<<1, THREADS_PER_BLOCK>>>(n,
                                                        c, 
                                                        ptr(V),
                                                        ptr(N),
                                                        ptr(W),
                                                        ptr(C),
                                                        ptr(uniqueC),
                                                        ptr(ac),
                                                        wm,
                                                        ptr(dQc));
            Qc = dQc[0];
            
            cerr << "modularity: " << Qc << endl;

        } while (abs(Qc - Qp) > threshold);
        

        //AGGREGATION PHASE

        //maybe it is possible to merge this kernels?
        degree = dvi(n, 0); 
        initializeDegreeGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(V), ptr(W), ptr(degree));

        dvi comDegree(n, 0);
        initializeComDegreeGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(degree), ptr(C), ptr(comDegree));

        comSize = dvi(n, 0);
        initializeComSizeGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(C), ptr(comSize));
    
        dvi newID(n, 0);
        initializeNewIDGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(C), ptr(comSize), ptr(newID));
        thrust_inclusive_scan(newID);
        thrust_sub_for_each(newID, 1);

        dvi edgePos = comDegree;
        thrust_inclusive_scan(edgePos);

        dvi vertexStart = comSize;
        thrust_inclusive_scan(vertexStart);

        dvi comm(n);
        initializeCommGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(C), ptr(comm), ptr(vertexStart));

        //merge community
        //new graph
        int newn; 
        int newm; 
        dvi newV;
        dvi newN; 
        dvf newW;

        newn = newID.back() + 1;
        newm = edgePos.back();

        newV = dvi(newn + 1, 0);
        initializeNewVGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(C), ptr(newID), ptr(edgePos), ptr(newV));

      
        newN = dvi(newm, -1);
        newW = dvf(newm, NO_EDGE);

      
        dvi hashSize = dvi(newn);
        thrust_copy_if_non_zero(comDegree, comSize, hashSize);   

        dvi hashOffset;
        dvi hashComm;
        dvf hashWeight;
        thrust_transform_hashmap_size(hashSize, hashSize, 1.5);
        hashMapCreate(hashSize, hashOffset, hashComm, hashWeight);
       
        mergeCommunityFillHashMapGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, 
                                                                            ptr(V), 
                                                                            ptr(N), 
                                                                            ptr(W), 
                                                                            ptr(C), 
                                                                            ptr(comm), 
                                                                            ptr(degree), 
                                                                            ptr(newID), 
                                                                            ptr(hashOffset), 
                                                                            ptr(hashComm), 
                                                                            ptr(hashWeight),
                                                                            DEBUG);

        mergeCommunityInitializeGraphGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(ptr(hashOffset), 
                                                                                ptr(hashComm),
                                                                                ptr(hashWeight), 
                                                                                newn, 
                                                                                ptr(newV), 
                                                                                ptr(newN), 
                                                                                ptr(newW));
        
        saveFinalCommunitiesGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(initialN, ptr(finalC), ptr(C), ptr(newID));

        //update graph
        n = newn; 
        m = newm; 
        V = newV;
        N = newN; 
        W = newW;
    } while (abs(Qc - Qba)> threshold);

    cout << fixed << Qc << endl;

    float elapsedTime = stopRecordingTime(startTime, stopTime);
    printf("%3.1f ms\n", elapsedTime);

    if (showAssignment) {
        hvi host_finalC = finalC;
        vi tmpFinalC(ptr(host_finalC), ptr(host_finalC) + initialN); 
        printClustering(initialN, tmpFinalC);
    }
    return 0;
}
