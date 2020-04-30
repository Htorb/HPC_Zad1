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
#include <thrust/unique.h>
#include <thrust/transform.h>
#include <thrust/functional.h>






//ASSUMPTIONS 
// there is at least one edge with a positive weight


// TODO bigger threshold int the first iteration
// TODO bucket partition
// TODO two hashmaps; hash function
// TODO how can update modularity
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
    uniqueC = duniqueC; \
    comSize = dcomSize; \
    finalC = dfinalC; 

#define NO_EDGE 0
#define BLOCKS_NUMBER 16
#define THREADS_PER_BLOCK 128
#define EMPTY_SLOT -1

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
    __host__ __device__ int operator()(const int &x) const {
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

//size > 1 !!
int h1(int Ci) {
    return 7 * Ci + 5;
}

int h2(int Ci) {
    return 2 * Ci + 1; 
}
int doubleHash(int Ci, int itr, int size) { 
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




  
__device__ void modularityArgMax(int* maxCj, float* maxDeltaAlmostMod, int pos, int i, int ci, float* hashWeight, int* hashComm, float* k, float* ac, int* comSize, float wm) {
    int cj = hashComm[pos];
    float wsum = hashWeight[pos];
 
    float deltaAlmostMod = wsum / wm 
    + k[i] * (ac[ci] - k[i] - ac[cj]) / (2 * wm * wm);


    if (deltaAlmostMod > *maxDeltaAlmostMod || deltaAlmostMod == *maxDeltaAlmostMod && cj < *maxCj) {
        if (comSize[cj] > 1 || comSize[ci] > 1 || cj < ci) {
            *maxCj = cj;
            *maxDeltaAlmostMod = deltaAlmostMod;
        }
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
    __shared__ int partials[THREADS_PER_BLOCK];    
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        int tid = threadIdx.x;
        int step = blockDim.x;
        int offset = hashOffset[i];
        int size = hashOffset[i + 1] - offset;
        int ci = C[i];
        int itr, pos;

        if (size == 0) {
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
                    if (EMPTY_SLOT != atomicCAS(&hashComm[pos], EMPTY_SLOT, cj)) {
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
        partialDeltaAlmostMod[tid] = -1;
        for (int j = offset + tid; j < offset + size; j += step) {
            if (hashComm[j] == EMPTY_SLOT)
                continue;
            modularityArgMax(&partialCMax[tid], &partialDeltaAlmostMod[tid], j, i, ci, hashWeight, hashComm, k, ac, comSize, wm);            
        }
        __syncthreads();

        for (int s = blockDim.x / 2; s >0 ; s >>= 1) {
            if (tid < s) { //TODO
                if (deltaAlmostMod > *maxDeltaAlmostMod || deltaAlmostMod == *maxDeltaAlmostMod && cj < *maxCj) {
                    if (comSize[cj] > 1 || comSize[ci] > 1 || cj < ci) {
                        *maxCj = cj;
                        *maxDeltaAlmostMod = deltaAlmostMod;
                    }
                }   
            }
            __syncthreads();
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
        __syncthreads();
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
                Q += W[j] / (2 * wm);
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

void initializeNewID(int n, const hvi& C, const hvi& comSize, hvi& newID) {
    for (int i = 0; i < n; ++i) {
        if (comSize[C[i]] != 0) {
            newID[C[i]] = 1;
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

    do { 
        TODEVICE
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
        TOHOST
        do {
            TODEVICE
            dnewComm = dC;
            dcomSize = dvi(n, 0);
            initializeComSizeGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(dC), ptr(dcomSize));
            ddegree = dvi(n, 0);
            initializeDegreeGPU<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(dV), ptr(dW), ptr(ddegree));

            
            
            hashMapSizesGenerator getHashMapSize(1.5);
            dvi dhashSize = dvi(n);
            thrust::transform(ddegree.begin(), ddegree.end(), dhashSize.begin(), getHashMapSize);
            
            dvi dhashOffset = dvi(n + 1);
            dhashOffset[0] = 0;
            thrust::inclusive_scan(dhashSize.begin(), dhashSize.end(), dhashOffset.begin() + 1);

            dvf dhashWeight(dhashOffset.back(), 0);
            dvi dhashComm(dhashOffset.back(), EMPTY_SLOT);

            
            hvi hashOffset = dhashOffset;
            hvf hashWeight = dhashWeight;
            hvi hashComm = dhashComm;
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
            TOHOST
            C = newComm;

            ac.assign(n, 0);
            initializeAc(n, C, k, ac);
            dassert(abs(sum(ac) - 2 * wm) < 0.0001);

            Qp = Qc;
            initializeUniqueCAndC(n, C, uniqueC, c);
            Qc = calculateModularity(n, c, V, N, W, C, uniqueC, ac, wm);
            
            cerr << "modularity: " << Qc << endl;
            if (DEBUG) {
                pvec(C);
                pvec(k);
                pvec(ac);
                pvec(comSize);
            }

        } while (abs(Qc - Qp) > threshold);

        //aggregation phase
        comSize = hvi(n, 0);
        hvi comDegree(n, 0);

        degree = hvi(n, 0); //check if needed
        initializeDegree(n, V, W, degree);

        initializeComSize(n, C, comSize);
        initializeComDegree(n, degree, C, comDegree);

        hvi newID(n, 0);

        initializeNewID(n, C, comSize, newID);
       
        cumsum(newID);

        hvi edgePos = comDegree;
        cumsum(edgePos);

        hvi vertexStart = comSize;
        cumsum(vertexStart);

        hvi comm(n, 0);
        initializeComm(n, C, comm, vertexStart);

         
        //merge community
        //new graph
        int newn; 
        int newm; 
        hvi newV;
        hvi newN; 
        hvf newW;

        newn = newID.back();
        newm = edgePos.back();

        newV = hvi(newn + 1, 0);
        for (int i = 0; i < n; ++i) {
            newV[newID[C[i]]] = edgePos[C[i]];
        }

        newN = hvi(newm, -1);
        newW = hvf(newm, NO_EDGE);

        map<int, float> hashMap;
        int oldc = C[comm[0]]; //can be n = 0?
        for (int idx = 0; idx < n; ++idx) {
            int i = comm[idx];
            int ci = C[i];
            if (oldc != ci) {
                int edgeId = newV[newID[oldc] - 1];
                for (auto it = hashMap.begin(); it != hashMap.end(); it++ ) {
                    float cj = it->first;
                    float wsum = it->second;
                    newN[edgeId] = newID[cj] - 1;
                    newW[edgeId] = wsum;
                    edgeId++;
                }
                oldc = C[comm[idx]];
                hashMap.clear();
            } 
            
            for (int j = V[i]; j < V[i + 1]; ++j) {
                if (W[j] == NO_EDGE)
                    break; 
                int cj = C[N[j]];
                if (hashMap.count(cj) == 0) {
                    hashMap[cj] = 0;
                }
                hashMap[cj] += W[j];
            }
            
        }

        int edgeId = newV[newID[oldc] - 1];
        for (auto it = hashMap.begin(); it != hashMap.end(); it++ ) {
            float cj = it->first;
            float wsum = it->second;
            newN[edgeId] = newID[cj] - 1;
            newW[edgeId] = wsum;
            edgeId++;
        }


        for (int i = 0; i < initialN; ++i) {
            finalC[i] = newID[C[finalC[i]]] - 1;
        }
        
        if (DEBUG) {
            cerr << "sum of weights: " << positiveSum(W) << endl;
        }

        //update graph
        n = newn; 
        m = newm; 
        V = newV;
        N = newN; 
        W = newW;
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
        printClustering(initialN, finalC);
    }
    return 0;
}
