#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>  
#include <vector>
#include <map>
#include <set>
#include <cassert> 
#include <chrono> 
#include "helpers.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/transform.h>
#include <thrust/functional.h>



//ASSUMPTIONS 
// there is at least one edge with a positive weight

// TODO bucket partition
// TODO how can update modularity
// TODO code grooming
// TODO correct outer loop condition (from slack)


#define dassert(a) \
    if ((DEBUG)) assert((a));

#define pvec(a) \
    do { std::cerr << #a << ": "; printVec((a)) ; } while(false)

#define ptr(a) \
    thrust::raw_pointer_cast((a).data())

#define TODEVICE \
    dV = V; \
    dN = N; \
    dW = W; \
    dC = C; \
    dnewComm = newComm; \
    dk = k; \
    dac = ac; \
    ddegree = degree; \
    duniqueC = uniqueC; \
    dcomSize = comSize; \
    dfinalC = finalC; 

#define TOHOST \
    V = dV; \
    N = dN; \
    W = dW; \
    C = dC; \
    newComm = dnewComm; \
    k = dk; \
    ac = dac; \
    degree = ddegree; \
    uniqueC = duniqueC; \
    comSize = dcomSize; \
    finalC = dfinalC; 

#define NO_EDGE 0
#define BLOCKS_NUMBER 16
#define THREADS_PER_BLOCK 128
#define EMPTY_SLOT -1

using namespace thrust::placeholders;
using namespace std;

using pi = pair<int, int>;
using tr = pair<pi, float>;
using hvi = thrust::host_vector<int>;
using hvf = thrust::host_vector<float>;
using dvi = thrust::device_vector<int>;
using dvf = thrust::device_vector<float>;



//float ITR_MODULARITY_THRESHOLD = 0.1;
bool DEBUG = false;



template<typename T>
void head(thrust::host_vector<T> v, int n = 5) {
    for (int i = 0; i < min(n, (int) v.size()); i++) {
         cerr << v[i] << " ";
    }
    cerr << endl;
}

template<typename T>
void printVec(thrust::host_vector<T> v){
    head(v, v.size());
}


template<typename T>
T sum(thrust::host_vector<T> v) {
    T sum = 0;
    for (auto val : v) {
        sum += val;
    }
    return sum;
}

template<typename T>
T positiveSum(thrust::host_vector<T> v) {
    T sum = 0;
    for (auto val : v) {
        if (val > 0) {
            sum += val;
        }
    }
    return sum;
}

template<typename T>
void cumsum(thrust::host_vector<T>& v) {
    int sum = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i];
        v[i] = sum;
    }
}

void parseCommandline(bool& showAssignment,
                        float& threshold,
                        string& matrixFile,
                        int argc,
                        char** argv) {
    int i = 1;
    while (i < argc) {
        string s(argv[i]);
        if (s == "-f") {
            matrixFile = string(argv[i + 1]);
            i += 2;
        } else if (s == "-g") {
            threshold = strtof(argv[i + 1], NULL);
            i += 2;
        } else if (s == "-v") {
            showAssignment = true;
            i += 1;
        }
        else if (s == "-d") {
            DEBUG = true;
            i += 1;
        } else {
            exit(1);
        }
    }
}

void printClustering(int initialN, const hvi& finalC) {
    vector<pi> finalCPrime;
    for (int i = 0; i < initialN; ++i) {
        finalCPrime.push_back(pi(finalC[i], i));
    }
    
    cout << std::set<int>(finalC.begin(), finalC.end()).size();
    sort(finalCPrime.begin(), finalCPrime.end());
    
    int lastC = -1;
    for (auto& p : finalCPrime) {
        if (lastC != p.first) {
            cout << endl << p.first;
            lastC = p.first;
        }
        cout << " " << p.second;
    }
    cout << endl;
}

void readGraphFromFile(const string& matrixFile, 
                        int& n,
                        int& m,
                        hvi& V, 
                        hvi& N,
                        hvf& W) {
    ifstream matrixStream;
    matrixStream.open(matrixFile);
    int entries = 0;
    matrixStream >> n >> n >> entries;
    
    m = 0;
    vector<tr> tmp;
    for (int i = 0; i < entries; i++) {
        int v1, v2;
        float f;
        matrixStream >> v1 >> v2 >> f;

        m++;
        tmp.push_back(tr(pi(v1 - 1,v2 - 1),f));
        //if graph is undirected
        if (v1 != v2) {
            m++;
            tmp.push_back(tr(pi(v2 - 1,v1 - 1),f));
        }
    }

    sort(tmp.begin(), tmp.end());

    V = hvi(n + 1, 0);
    N = hvi(m, 0);
    W = hvf(m, 0);

    int v_idx = 0;
    for (size_t i = 0; i < tmp.size(); i++) {
        while (v_idx <= tmp[i].first.first) {
            V[v_idx++] = i;
        }
        N[i] = tmp[i].first.second;
        W[i] = tmp[i].second;
    }
    while (v_idx < n + 1) {
        V[v_idx++] = m;
    }
}




////////////////////////////////////////////////////////////////////////////////////////////
struct hashMapSizesGenerator{
    const float ratio;
    hashMapSizesGenerator(float _ratio) : ratio(_ratio) {}

    // vector<int> computePrimeNumbers(int upTo) {
    //     vector<int> primes;
    //     for (int i = 2; i <= upTo; ++i) {
    //         bool isPrime = true;
    //         for (auto p : primes) {
    //             if (i % p == 0) {
    //                 isPrime = false;
    //                 break; 
    //             }
    //         }
    //         if (isPrime) {
    //             primes.push_back(i);
    //         }
    //     }
    //     return primes;
    // }
    __host__ __device__ 
   int operator()(const int &x) const {
        if (x == 0) return 0;
        int v = (int) (ratio * (x + 1));
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }
};

struct isNotZero {
  __host__ __device__
  bool operator()(const int &x) const {
    return x != 0;
  }
};

__host__  __device__ int h1(int Ci) {
    return 7 * Ci + 5;
}

__host__  __device__ int h2(int Ci) {
    return 2 * Ci + 1; 
}
//size > 0 !!
__host__  __device__ int doubleHash(int Ci, int itr, int size) { 
    return (h1(Ci) + itr * h2(Ci)) % size; //maybe can be improved
}


void computeMove(int i,
                    int n,
                    hvi& newComm, 
                    const hvi& V,
                    const hvi& N, 
                    const hvf& W, 
                    const hvi& C,
                    const hvi& comSize, 
                    const hvf& k, 
                    const hvf& ac, 
                    const float wm,
                    hvi& hashOffset,
                    hvf& hashWeight,
                    hvi& hashComm
                ) {
    int offset = hashOffset[i];
    int size = hashOffset[i + 1] - offset;
    int ci = C[i];
    int itr, pos;

    if (size == 0) {
        newComm[i] = ci;
        return;
    }

    for (int j = V[i]; j < V[i + 1]; ++j) {
        if (W[j] == NO_EDGE)
            break;

        int cj = C[N[j]];
        itr = 0;
        do {
            // cerr << "szukam hasha" << endl;
            pos = offset + doubleHash(cj, itr, size);
            // cerr << "znalazlem!" << endl;

            // cerr << "ci: " << ci << " itr: " << " cj: " << cj << " itr: " << itr << " size: " << size << " offset: " << offset << " pos: " << pos << endl;

            if (hashComm[pos] == cj) {
                if (N[j] != i) {
                    hashWeight[pos] += W[j]; 
                }
            } else if (hashComm[pos] == EMPTY_SLOT) {
                hashComm[pos] = cj;
                if (N[j] != i) {
                    hashWeight[pos] += W[j]; 
                }
            }
            itr++;
        } while (hashComm[pos] != cj);
    }

    int maxCj = n;
    float maxDeltaAlmostMod = -1;

    for (pos = offset; pos < offset + size; pos++) {
        // cerr << "kurdebele" << endl;
        int cj = hashComm[pos];
        if (cj == EMPTY_SLOT)
            continue;
        float wsum = hashWeight[pos];

        float deltaAlmostMod = wsum / wm 
            + k[i] * (ac[ci] - k[i] - ac[cj]) / (2 * wm * wm);
        // cerr << "node: " << i << " to: " << cj << " deltaAlmostMod: " << deltaAlmostMod << endl; 

        if (deltaAlmostMod > maxDeltaAlmostMod || deltaAlmostMod == maxDeltaAlmostMod && cj < maxCj) {
            if (comSize[cj] > 1 || comSize[ci] > 1 || cj < ci) {
                maxCj = cj;
                maxDeltaAlmostMod = deltaAlmostMod;
            }
        }   
    }

    itr = 0;
    do {    
        // cerr << "szukam hasha" << endl;
        pos = offset + doubleHash(ci, itr, size);
        // cerr << "znalazlem!" << endl;
        // cerr << "ci: " << ci << " size: " << size << " itr: " << itr << " pos: " << pos << " hashComm[pos]: " << hashComm[pos] << endl; 

        itr++;
    } while (hashComm[pos] != ci && hashComm[pos] != EMPTY_SLOT);

    //if not found better move maxDeltaMod will be negative
    float maxDeltaMod = maxDeltaAlmostMod - hashWeight[pos] / wm;
    //cerr << "node: " << i << " to: " << maxCj << " maxDeltaMod: " << maxDeltaMod << " hashMap[ci]: " << hashMap[ci] << endl; 
    // cerr << "eeee" << endl;

    if (maxDeltaMod > 0) {
        newComm[i] = maxCj;
    } else {
        newComm[i] = ci;
    }
    // cerr << "elko" << endl;
}



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
        int itr, pos;

        if (size == 0) {
            if (tid == 0)
                newComm[i] = ci;
            continue;
        }

        for (int j = tid + V[i]; j < V[i + 1]; j += step) {
            if (W[j] == NO_EDGE)
                break;

            int cj = C[N[j]];
            itr = 0;
            do {
                // cerr << "szukam hasha" << endl;
                pos = offset + doubleHash(cj, itr, size);
                // cerr << "znalazlem!" << endl;

                // cerr << "ci: " << ci << " itr: " << " cj: " << cj << " itr: " << itr << " size: " << size << " offset: " << offset << " pos: " << pos << endl;

                if (hashComm[pos] == cj) {
                    if (N[j] != i) {
                        atomicAdd(&hashWeight[pos], W[j]); 
                    }
                } else if (hashComm[pos] == EMPTY_SLOT) {
                    if (cj == atomicCAS(&hashComm[pos], EMPTY_SLOT, cj)) {
                        if (N[j] != i) {
                            atomicAdd(&hashWeight[pos], W[j]); 
                        }
                    } 
                    else if (hashComm[pos] == cj) {
                        if (N[j] != i) {
                            atomicAdd(&hashWeight[pos], W[j]); 
                        }
                    }
                }
                itr++;
            } while (hashComm[pos] != cj);
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
            itr = 0;
            do {    
                // cerr << "szukam hasha" << endl;
                pos = offset + doubleHash(ci, itr, size);
                // cerr << "znalazlem!" << endl;
                // cerr << "ci: " << ci << " size: " << size << " itr: " << itr << " pos: " << pos << " hashComm[pos]: " << hashComm[pos] << endl; 

                itr++;
            } while (hashComm[pos] != ci && hashComm[pos] != EMPTY_SLOT);

            //if not found better move maxDeltaMod will be negative
            //cerr << "node: " << i << " to: " << maxCj << " maxDeltaMod: " << maxDeltaMod << " hashMap[ci]: " << hashMap[ci] << endl; 
            // cerr << "eeee" << endl;

            if (partialDeltaMod[0] - hashWeight[pos] / wm > 0) {
                newComm[i] = partialCMax[0];
            } else {
                newComm[i] = ci;
            }
        }
    // cerr << "elko" << endl;
    }
}

float calculateModularity(  int n,
                            int c,
                            const hvi& V,
                            const hvi& N, 
                            const hvf& W, 
                            const hvi& C,
                            const hvi& uniqueC, 
                            const hvf& ac, 
                            const float wm) {
    float Q = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = V[i]; j < V[i + 1]; ++j) {
            if (W[j] == NO_EDGE)
                break; 
            if (C[N[j]] == C[i]) {
                Q += W[j] / 2 / wm;
            }
        }
    }

    for (int i = 0; i < c; ++i) {
        Q -= ac[uniqueC[i]] * ac[uniqueC[i]] / (4 * wm * wm);
    }
    return Q;
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

void initializeCommunities(int n, hvi& C) {
    for (int i = 0; i < n; ++i) {
            C[i] = i;
    }
}

__global__ void initializeK(int n, const int* V, const float* W, float* k) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        for (int j = V[i]; j < V[i + 1]; ++j) {
            if (W[j] == NO_EDGE)
                break;
            k[i] += W[j];
        } 
    }
}

void initializeAc(int n, const hvi& C, const hvf& k, hvf& ac) {
    for (int i = 0; i < n; ++i) {
        ac[C[i]] += k[i]; //attomic add
    }
}

__global__ void initializeAcGPU(int n, int* C, float* k, float* ac) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&ac[C[i]], k[i]);
    }
}

void initializeUniqueCAndC(int n, const hvi& C, hvi& uniqueC, int& c) {
    uniqueC = C;
    set<int> s(uniqueC.begin(), uniqueC.end());
    uniqueC.assign(s.begin(), s.end());
    c = s.size();
}

void initializeUniqueCAndCGPU(int n, const dvi& C, dvi& uniqueC, int& c) {
    uniqueC = C;
    thrust::sort(uniqueC.begin(), uniqueC.end());
    thrust::unique(uniqueC.begin(), uniqueC.end());
    c = uniqueC.size();
}
        

void initializeDegree(int n, const hvi& V, const hvf& W, hvi& degree) {
    for (int i = 0; i < n; ++i) {
        int ctr = 0;
        for (int j = V[i]; j < V[i + 1]; ++j) {
            if (W[j] == NO_EDGE)
                break;
            ctr++;
        }
        degree[i] = ctr;
    }
}

__global__ void initializeDegreeGPU(int n, int* V, float* W, int* degree) {
    for (int i = 0; i < n; ++i) {
        int ctr = 0;
        for (int j = V[i]; j < V[i + 1]; ++j) {
            if (W[j] == NO_EDGE)
                break;
            ctr++;
        }
        degree[i] = ctr;
    }
}


void initializeComSize(int n, const hvi& C, hvi& comSize) {
    for (int i = 0; i < n; ++i) {
        comSize[C[i]] += 1; //atomic
    }
}

__global__ void initializeComSizeGPU(int n, int* C, int* comSize) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&comSize[C[i]], 1);
    }
}


void initializeComDegree(int n, const hvi& degree, const hvi& C, hvi& comDegree) {
    for (int i = 0; i < n; ++i) {
        comDegree[C[i]] += degree[i]; //atomic
    }
}

__global__ void initializeComDegreeGPU(int n, int* degree, int* C, int* comDegree) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&comDegree[C[i]], degree[i]); 
    }
}

void initializeNewID(int n, const hvi& C, const hvi& comSize, hvi& newID) {
    for (int i = 0; i < n; ++i) {
        if (comSize[C[i]] != 0) {
            newID[C[i]] = 1;
        }
    }
}

__global__ void initializeNewIDGPU(int n, int* C, int* comSize, int* newID) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (comSize[C[i]] != 0) {
            atomicCAS(&newID[C[i]], 0, 1);
        }
    }
}

void initializeComm(int n, const hvi& C, hvi& comm, hvi& vertexStart) {
    for (int i = 0; i < n; ++i) {
        vertexStart[C[i]] -= 1; //in paper is add, atomic
        int res = vertexStart[C[i]];
        comm[res] = i; 
    }
}
__global__ void initializeCommGPU(int n, int* C, int* comm, int* vertexStart) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int res = atomicSub(&vertexStart[C[i]], 1) - 1;
        comm[res] = i; 
    }
}

void initializeNewV(int n, const hvi& C,  const hvi& newID, const hvi& edgePos, hvi& newV) {
    for (int i = 0; i < n; ++i) {
        newV[newID[C[i]] + 1] = edgePos[C[i]];
    }
}

__global__ void initializeNewVGPU(int n, int* C, int* newID, int* edgePos, int* newV) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicCAS(&newV[newID[C[i]] + 1], 0, edgePos[C[i]]);
    }
}

void mergeCommunity(int n,
                    const hvi& V,
                    const hvi& N,
                    const hvf& W,
                    const hvi& C,
                    const hvi& comm,
                    const hvi& degree,
                    const hvi& newID,
                    const hvi& hashOffset,
                    hvi& hashComm,
                    hvf& hashWeight,
                    int newn,
                    const hvi& newV,
                    hvi& newN,
                    hvf& newW) {
    for (int idx = 0; idx < n; ++idx) {
        int i = comm[idx];
        int newci = newID[C[i]];
        int offset = hashOffset[newci];
        int size = hashOffset[newci + 1] - offset;
        int itr, pos;

        if (size == 0) {
            continue;
        }

        if (DEBUG) {
            assert(size >= degree[i]);
        }

        for (int j = V[i]; j < V[i + 1]; ++j) {
            if (W[j] == NO_EDGE)
                break; 

            int newcj = newID[C[N[j]]];

            itr = 0;
            do {
                pos = offset + doubleHash(newcj, itr, size);
    
                if (hashComm[pos] == newcj) {
                    hashWeight[pos] += W[j]; 
                } else if (hashComm[pos] == EMPTY_SLOT) {
                    hashComm[pos] = newcj;
                    hashWeight[pos] += W[j]; 
                }
                itr++;
            } while (hashComm[pos] != newcj);
        }   
    }

    for (int i = 0; i < newn; ++i) {
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

void saveFinalCommunities(int initialN,
                          hvi& finalC,
                          const hvi& C,
                          const hvi& newID) {
    for (int i = 0; i < initialN; ++i) {
        finalC[i] = newID[C[finalC[i]]];
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
        int itr, pos;

        if (size == 0) {
            continue;
        }

        if (DEBUG) {
            assert(size >= degree[i]);
        }

        for (int j = tid + V[i]; j < V[i + 1]; j += step) {
            if (W[j] == NO_EDGE)
                break; 

            int newcj = newID[C[N[j]]];

            itr = 0;
            do {
                pos = offset + doubleHash(newcj, itr, size);
    
                if (hashComm[pos] == newcj) {
                        atomicAdd(&hashWeight[pos], W[j]); 
                } else if (hashComm[pos] == EMPTY_SLOT) {
                    if (newcj == atomicCAS(&hashComm[pos], EMPTY_SLOT, newcj)) {
                            atomicAdd(&hashWeight[pos], W[j]); 
                    } 
                    else if (hashComm[pos] == newcj) {
                            atomicAdd(&hashWeight[pos], W[j]); 
                    }
                }
                itr++;
            } while (hashComm[pos] != newcj);
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
    hvi V; //vertices
    hvi N; //neighbours
    hvf W; //weights
    float wm; //sum of weights
    hvi C; //current clustering
    hvi newComm; //temporary array to store new communities
    hvf k; //sum of vertex's edges
    hvf ac; //sum of cluster edges
    int c; //number of communities
    hvi uniqueC; //list of unique communities ids
    hvi comSize; //size of ech community
    hvi degree; //degree of each vertex

    

    int initialN; //number of vertices in the first iteration
    hvi finalC; //final clustering result 

    //graph vars device
    dvi dV; 
    dvi dN; 
    dvf dW;
    dvi dC; 
    dvi dnewComm;
    dvf dk; 
    dvf dac;
    dvi duniqueC; 
    dvi dcomSize; 
    dvi dfinalC;
    dvi ddegree;

    hashMapSizesGenerator getHashMapSize(1.5);

    float Qba, Qp, Qc; //modularity before outermostloop iteration, before and after modularity optimisation respectively
    
    cudaEvent_t startTime, stopTime;
    float elapsedTime;


    parseCommandline(showAssignment, threshold, matrixFile, argc, argv);
    readGraphFromFile(matrixFile, n, m, V, N, W);

    HANDLE_ERROR(cudaEventCreate(&startTime));
    HANDLE_ERROR(cudaEventCreate(&stopTime));
    HANDLE_ERROR(cudaEventRecord(startTime, 0));

    initialN = n;
    wm = sum(W) / 2;

    finalC = hvi(n, 0);
    initializeCommunities(initialN, finalC);

    TODEVICE
    do { 
        dC = dvi(n, 0);//redundant?
        thrust::sequence(dC.begin(), dC.end()); //initializeCommunities

        dk = dvf(n, 0);
        initializeK<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, 
                                                          ptr(dV), 
                                                          ptr(dW), 
                                                          ptr(dk)); 

        
        float ksum = thrust::reduce(dk.begin(), dk.end(), (float) 0, thrust::plus<float>());
        dassert(abs(ksum - 2 * wm) < 0.0001);

        dac = dk; 
        

        //modularity optimisation phase

        //
        initializeUniqueCAndCGPU(n, dC, duniqueC, c);


        // Qc = calculateModularity(n, c, V, N, W, C, uniqueC, ac, wm);
        dvf dQc(1);
        calculateModularityGPU<<<1, THREADS_PER_BLOCK>>>(n,
                                                        c, 
                                                        ptr(dV),
                                                        ptr(dN),
                                                        ptr(dW),
                                                        ptr(dC),
                                                        ptr(duniqueC),
                                                        ptr(dac),
                                                        wm,
                                                        ptr(dQc));
        Qc = dQc[0];
        Qba = Qc;

        
        cerr << "modularity: " << Qc << endl;
        if (DEBUG) {
            pvec(C);
            pvec(k);
            pvec(ac);
            pvec(comSize);
        }
        do {
            
            dnewComm = dC;
            dcomSize = dvi(n, 0);
            initializeComSizeGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(dC), ptr(dcomSize));
            ddegree = dvi(n, 0);
            initializeDegreeGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(dV), ptr(dW), ptr(ddegree));

            
            
            dvi dhashSize = dvi(n);
            thrust::transform(ddegree.begin(), ddegree.end(), dhashSize.begin(), getHashMapSize);
            
            dvi dhashOffset = dvi(n + 1); //TODO move memory allocation up
            dhashOffset[0] = 0;
            thrust::inclusive_scan(dhashSize.begin(), dhashSize.end(), dhashOffset.begin() + 1);

            dvf dhashWeight(dhashOffset.back(), 0);
            dvi dhashComm(dhashOffset.back(), EMPTY_SLOT);
            
          
            computeMoveGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, 
                                                                 ptr(dnewComm), 
                                                                 ptr(dV), 
                                                                 ptr(dN), 
                                                                 ptr(dW), 
                                                                 ptr(dC), 
                                                                 ptr(dcomSize), 
                                                                 ptr(dk), 
                                                                 ptr(dac), 
                                                                 wm, 
                                                                 ptr(dhashOffset), 
                                                                 ptr(dhashWeight), 
                                                                 ptr(dhashComm));
            
            if (DEBUG) {
                TOHOST
                //compute move on cpu and gpu work the same way                         
                float dhashsum = thrust::reduce(dhashWeight.begin(), dhashWeight.end(), (float) 0, thrust::plus<float>());                                 
                

                newComm = C;
                hvi hashOffset = dhashOffset;
                hvf hashWeight(hashOffset.back(), 0);
                hvi hashComm(hashOffset.back(), EMPTY_SLOT);
                // hvf hashWeight = dhashWeight;
                // hvi hashComm = dhashComm;
                for (int i = 0; i < n; ++i) {	            
                    computeMove(i, n, newComm, V, N, W, C, comSize, k, ac, wm, hashOffset, hashWeight, hashComm);
                }
                
                float hashsum = thrust::reduce(hashWeight.begin(), hashWeight.end(), (float) 0, thrust::plus<float>());                                 
                assert(abs(dhashsum - hashsum) < 0.001);
            }
            
            dC = dnewComm;

            dac.assign(n, 0);
            initializeAcGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(dC), ptr(dk), ptr(dac));
            if (DEBUG) {
                float acsum = thrust::reduce(dac.begin(), dac.end(), (float) 0, thrust::plus<float>());                                 
                assert(abs(acsum - 2 * wm) < 0.0001);
            }
            

            Qp = Qc;
            initializeUniqueCAndCGPU(n, dC, duniqueC, c);
            dvf dQc(1);
            calculateModularityGPU<<<1, THREADS_PER_BLOCK>>>(n,
                                                        c, 
                                                        ptr(dV),
                                                        ptr(dN),
                                                        ptr(dW),
                                                        ptr(dC),
                                                        ptr(duniqueC),
                                                        ptr(dac),
                                                        wm,
                                                        ptr(dQc));
            Qc = dQc[0];
            
            cerr << "modularity: " << Qc << endl;
            if (DEBUG) {
                pvec(C);
                pvec(k);
                pvec(ac);
                pvec(comSize);
            }

        } while (abs(Qc - Qp) > threshold);
        

        //AGGREGATION PHASE

        //maybe it is possible to merge this kernels?
        ddegree = dvi(n, 0); 
        initializeDegreeGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(dV), ptr(dW), ptr(ddegree));

        dvi dcomDegree(n, 0);
        initializeComDegreeGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(ddegree), ptr(dC), ptr(dcomDegree));

        dcomSize = dvi(n, 0);
        initializeComSizeGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(dC), ptr(dcomSize));
    
        dvi dnewID(n, 0);
        initializeNewIDGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(dC), ptr(dcomSize), ptr(dnewID));
        thrust::inclusive_scan(dnewID.begin(), dnewID.end(), dnewID.begin());
        thrust::for_each(dnewID.begin(), dnewID.end(), _1 -= 1);

        dvi dedgePos = dcomDegree;
        thrust::inclusive_scan(dedgePos.begin(), dedgePos.end(), dedgePos.begin());

        dvi dvertexStart = dcomSize;
        thrust::inclusive_scan(dvertexStart.begin(), dvertexStart.end(), dvertexStart.begin());

        dvi dcomm(n);
        initializeCommGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(dC), ptr(dcomm), ptr(dvertexStart));

        //merge community
        //new graph
        int newn; 
        int newm; 
        dvi dnewV;
        dvi dnewN; 
        dvf dnewW;

        newn = dnewID.back() + 1;
        newm = dedgePos.back();

        dnewV = dvi(newn + 1, 0);
        initializeNewVGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(dC), ptr(dnewID), ptr(dedgePos), ptr(dnewV));

      
        dnewN = dvi(newm, -1);
        dnewW = dvf(newm, NO_EDGE);

      

        dvi dhashSize = dvi(newn, 0);       //can use not newn but number of verstices with non zero degree
        thrust::copy_if(dcomDegree.begin(), dcomDegree.end(), dcomSize.begin(), dhashSize.begin(), isNotZero());
        // for (int i = 0; i < newn; i++) {
        //     cout << dhashSize[i] << " ";
        // }
        // cout << endl;
        thrust::transform(dhashSize.begin(), dhashSize.end(), dhashSize.begin(), getHashMapSize);
        

        dvi dhashOffset = dvi(newn + 1); //TODO move memory allocation up
        dhashOffset[0] = 0;
        thrust::inclusive_scan(dhashSize.begin(), dhashSize.end(), dhashOffset.begin() + 1);


        dvi dhashComm(dhashOffset.back(), EMPTY_SLOT);
        dvf dhashWeight(dhashOffset.back(), 0);

       

        // mergeCommunity(n, V, N, W, C, comm, degree, newID, hashOffset, hashComm, hashWeight, newn, newV, newN, newW);
        mergeCommunityFillHashMapGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, 
                                                                            ptr(dV), 
                                                                            ptr(dN), 
                                                                            ptr(dW), 
                                                                            ptr(dC), 
                                                                            ptr(dcomm), 
                                                                            ptr(ddegree), 
                                                                            ptr(dnewID), 
                                                                            ptr(dhashOffset), 
                                                                            ptr(dhashComm), 
                                                                            ptr(dhashWeight),
                                                                            DEBUG);
        mergeCommunityInitializeGraphGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(ptr(dhashOffset), 
                                                                                ptr(dhashComm),
                                                                                ptr(dhashWeight), 
                                                                                newn, 
                                                                                ptr(dnewV), 
                                                                                ptr(dnewN), 
                                                                                ptr(dnewW));
        

        saveFinalCommunitiesGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(initialN, ptr(dfinalC), ptr(dC), ptr(dnewID));

        

        //update graph
        n = newn; 
        m = newm; 
        dV = dnewV;
        dN = dnewN; 
        dW = dnewW;
    } while (abs(Qc - Qba)> threshold);

    auto endTime = chrono::steady_clock::now();
    
    // Store the time difference between start and end
    cout << fixed << Qc << endl;
    HANDLE_ERROR(cudaEventRecord(stopTime, 0));
    HANDLE_ERROR(cudaEventSynchronize(stopTime));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, startTime, stopTime));
    printf("%3.1f ms\n", elapsedTime);
    HANDLE_ERROR(cudaEventDestroy(startTime));
    HANDLE_ERROR(cudaEventDestroy(stopTime));

    if (showAssignment) {
        finalC = dfinalC;
        printClustering(initialN, finalC);
    }
    return 0;
}
