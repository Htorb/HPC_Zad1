#include <iostream>
#include <cstring>
#include <vector>
#include <map>
#include <set>
#include <cassert> 
#include <string>
#include <iomanip> //todo remove


#include "thrust_wrappers.h"
#include "utils.h"
#include "hashmap_utils.h"


// TODO bucket partition
// TODO correct outer loop condition (from slack)


#define ptr(a) thrust::raw_pointer_cast((a).data())

#define NO_EDGE 0

#define BLOCKS_NUMBER 16
#define THREADS_PER_BLOCK 128

struct Bucket {
    int from;
    int to;

    Bucket(int _from, int _to) : from(_from), to(_to) {};

    __host__ __device__
    bool operator()(const int &x) const {
        return from <= x && x <= to;
    }
};


struct StepPolicy {
    int node_start;
    int node_step;
    int edge_start;
    int edge_step;
};

 __device__ StepPolicy get_step_policy(int thread_id, int block_id, int block_dim, int grid_dim, int policy_nr) {
     switch (policy_nr) {
         case 0:
            StepPolicy sp;
            sp.node_start = block_id;
            sp.node_step = grid_dim;
            sp.edge_start = thread_id;
            sp.edge_step = block_dim;
            return sp;
            break;
     }
 }

__device__ void update_max_modularity(int* max_C, float* max_delta_modularity, int new_C, float new_delta_modulatiry) {
    if (new_delta_modulatiry > *max_delta_modularity || new_delta_modulatiry == *max_delta_modularity && new_C < *max_C) {
        *max_C = new_C;
        *max_delta_modularity = new_delta_modulatiry;
    }
}

__global__ void compute_move(int n,
                                int* new_C, 
                                int* V,
                                int* N, 
                                float* W, 
                                int* C,
                                int* comm_size, 
                                float* k, 
                                float* ac, 
                                const float weights_sum,
                                int* hash_offset,
                                float* hash_weight,
                                int* hash_comm,
                                int policy_nr) {
    __shared__ int partial_C_max[THREADS_PER_BLOCK];
    __shared__ float partial_delta_mod[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int step = blockDim.x;

    StepPolicy sp = get_step_policy(threadIdx.x, blockIdx.x, blockDim.x, gridDim.x, policy_nr);

    for (int i = sp.node_start; i < n; i += sp.node_step) {    
        int offset = hash_offset[i];
        int size = hash_offset[i + 1] - offset;
        int ci = C[i];
        int pos;

        if (size == 0) {
            continue;
        }

        for (int j = sp.edge_start + V[i]; j < V[i + 1]; j += sp.edge_step) {
            if (W[j] == NO_EDGE)
                break;
            if (N[j] != i) {
                hashmap_insert(hash_comm, hash_weight, offset, size, C[N[j]], W[j]);
            }
        }
        __syncthreads();

        partial_C_max[tid] = n;
        partial_delta_mod[tid] = -1;
        for (pos = offset + tid; pos < offset + size; pos += step) {
            if (hash_comm[pos] == EMPTY_SLOT)
                continue;

            int new_C = hash_comm[pos];
            
            float deltaMod = hash_weight[pos] / weights_sum 
                                + k[i] * (ac[ci] - k[i] - ac[new_C]) / 2 / weights_sum / weights_sum;
        
            if (comm_size[new_C] > 1 || comm_size[ci] > 1 || new_C < ci) {
                update_max_modularity(&partial_C_max[tid], &partial_delta_mod[tid], new_C, deltaMod);
            }
        }
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0 ; s >>= 1) {
            if (tid < s) {
                update_max_modularity(&partial_C_max[tid], &partial_delta_mod[tid], partial_C_max[tid + s], partial_delta_mod[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            pos = hashmap_find(hash_comm, offset, size, ci);

            if (partial_delta_mod[0] - hash_weight[pos] / weights_sum > 0) {
                new_C[i] = partial_C_max[0];
            } else {
                new_C[i] = ci;
            }
        }
    }
}

//WARNING WORKS ONLY WITH ONE KERNEL
__global__ void calculate_modularity(int n,
                                        int c,
                                        int* V,
                                        int* N, 
                                        float* W, 
                                        int* C,
                                        int* uniqueC, 
                                        float* ac, 
                                        const float weights_sum,
                                        float* Q) {
                                           
    __shared__ float partials[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim =  blockDim.x;
    int step = blockDim.x * gridDim.x;

    float a = 0;
    for (int i = bid * bdim + tid; i < n; i += step) {
        for (int j = V[i]; j < V[i + 1]; ++j) {
            if (C[N[j]] == C[i]) {
                a += W[j] / 2 / weights_sum;
                // printf("jestem: %d dodaje: %f\n", i , W[j] / 2 / weights_sum);
            }
        }
    }
    
    for (int i = bid * bdim + tid; i < c; i += step) {
        a -= ac[uniqueC[i]] * ac[uniqueC[i]] / 4 / weights_sum / weights_sum;
        // printf("jestem: %d odejmuje: %f\n", i , ac[uniqueC[i]] * ac[uniqueC[i]] / 4 / weights_sum / weights_sum);
    }
    partials[tid] = a;
    __syncthreads();

    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partials[tid] += partials[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        Q[bid] = partials[0];
    }
}

__global__ void initialize_k(int n, const int* V, const float* W, float* k) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        for (int j = V[i]; j < V[i + 1]; ++j) {
            if (W[j] == NO_EDGE)
                break;
            k[i] += W[j];
        } 
    }
}

__global__ void initialize_ac(int n, int* C, float* k, float* ac) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&ac[C[i]], k[i]);
    }
}

void initialize_uniqueC_and_C(int n, const dvi& C, dvi& uniqueC, int& c) {
    uniqueC = C;
    thrust_sort(uniqueC);
    c = thrust_unique(uniqueC);
}
        
__global__ void initialize_degree(int n, int* V, float* W, int* degree) {
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

__global__ void initialize_comm_size(int n, int* C, int* comm_size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&comm_size[C[i]], 1);
    }
}

__global__ void initialize_comm_degree(int n, int* degree, int* C, int* comDegree) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&comDegree[C[i]], degree[i]); 
    }
}

__global__ void initialize_newID(int n, int* C, int* comm_size, int* newID) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (comm_size[C[i]] != 0) {
            atomicCAS(&newID[C[i]], 0, 1);
        }
    }
}

__global__ void initialize_comm(int n, int* C, int* comm, int* vertex_start) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int res = atomicSub(&vertex_start[C[i]], 1) - 1;
        comm[res] = i; 
    }
}

__global__ void initialize_aggregated_V(int n, int* C, int* newID, int* edge_pos, int* aggregated_V) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicCAS(&aggregated_V[newID[C[i]] + 1], 0, edge_pos[C[i]]);
    }
}

__global__ void save_final_communities(int initial_n,
                                        int* finalC,
                                        int* C,
                                        int* newID) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < initial_n; i += blockDim.x * gridDim.x) {
        finalC[i] = newID[C[finalC[i]]];
    }
}

__global__ void merge_community_fill_hashmap(int n,
                                                int* V,
                                                int* N,
                                                float* W,
                                                int* C,
                                                int* comm,
                                                int* degree,
                                                int* newID,
                                                int* hash_offset,
                                                int* hash_comm,
                                                float* hash_weight,
                                                bool debug) {
    for (int idx = blockIdx.x; idx < n; idx += gridDim.x) {
        int tid = threadIdx.x;
        int step = blockDim.x;

        int i = comm[idx];
        int new_ci = newID[C[i]];
        int offset = hash_offset[new_ci];
        int size = hash_offset[new_ci + 1] - offset;

        if (size == 0) {
            continue;
        }

        if (debug) {
            assert(size >= degree[i]);
        }

        for (int j = tid + V[i]; j < V[i + 1]; j += step) {
            if (W[j] == NO_EDGE)
                break; 

            hashmap_insert(hash_comm, hash_weight, offset, size, newID[C[N[j]]], W[j]);
        }   
    }
}

__global__ void merge_community_initialize_graph(int* hash_offset,
                                                    int* hash_comm,
                                                    float* hash_weight,
                                                    int aggregated_n,
                                                    int* aggregated_V,
                                                    int* aggregated_N,
                                                    float* aggregated_W) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < aggregated_n; i += blockDim.x * gridDim.x) {
        int edgeId = aggregated_V[i];
        for (int pos = hash_offset[i]; pos < hash_offset[i + 1]; ++pos) {
            int new_cj = hash_comm[pos];
            if (new_cj == EMPTY_SLOT) {
                continue;
            }
            aggregated_N[edgeId] = new_cj;
            aggregated_W[edgeId] = hash_weight[pos];
            edgeId++;
        }        
    }
}


int main(int argc, char *argv[]) {
    //commandline vars
    bool show_assignment = false;
    float threshold = 0;
    std::string matrix_file;
    bool debug = false;

    int n; //number vertices 
    int m; //number of edges
    dvi V; //vertices
    dvi N; //neighbours
    dvf W; //weights
    float weights_sum; //sum of weights
    dvi C; //current clustering
    dvi new_C; //temporary array to store new communities
    dvf k; //sum of vertex's edges
    dvf ac; //sum of cluster edges
    int c; //number of communities
    dvi uniqueC; //list of unique communities ids
    dvi comm_size; //size of ech community
    dvi degree; //degree of each vertex

    int initial_n; //number of vertices in the first iteration
    dvi finalC; //final clustering result 

    float Qba, Qp, Qc; //modularity before outermostloop iteration, before and after modularity optimisation respectively
    
    cudaEvent_t start_time, stop_time;

    parse_command_line(show_assignment, threshold, matrix_file, argc, argv, debug);

    vi host_V;
    vi host_N;
    vf host_W;
    read_graph_from_file(matrix_file, n, m, host_V, host_N, host_W);
    V = host_V;
    N = host_N;
    W = host_W;

    start_recording_time(start_time, stop_time);
 
    initial_n = n;
    weights_sum = thrust_sum(W) / 2;

    finalC = dvi(n);
    thrust_sequence(finalC); 

    do { 
        C = dvi(n);
        thrust_sequence(C); 

        k = vf(n, 0);
        initialize_k<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(V), ptr(W), ptr(k)); 

            
        if (debug) {
            float ksum = thrust_sum(k);
            assert(abs(ksum - 2 * weights_sum) < 0.001 * ksum);
        }

        ac = k; 
        
        //modularity optimisation phase
        initialize_uniqueC_and_C(n, C, uniqueC, c);

        dvf dQc(BLOCKS_NUMBER);
        calculate_modularity<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, c, ptr(V), ptr(N), ptr(W), ptr(C), 
                                                        ptr(uniqueC), ptr(ac), weights_sum, ptr(dQc));
        Qc = thrust_sum(dQc);

        Qba = Qc;

        std::cerr << "modularity: " << Qc << std::endl;
        do {
            new_C = C;
            comm_size = dvi(n, 0);
            initialize_comm_size<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(C), ptr(comm_size));

            degree = dvi(n, 0);
            initialize_degree<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(V), ptr(W), ptr(degree));
            
            dvi hash_size = dvi(n);
            thrust_transform_hashmap_size(degree, hash_size, 1.5);
            


            dvi hash_offset;
            dvi hash_comm;
            dvf hash_weight;
            hashmap_create(hash_size, hash_offset, hash_comm, hash_weight);

            compute_move<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(new_C), ptr(V), ptr(N), ptr(W), 
                                                                 ptr(C), ptr(comm_size), ptr(k), ptr(ac), 
                                                                 weights_sum, ptr(hash_offset), ptr(hash_weight), ptr(hash_comm), 0);
            


            C = new_C;

            ac.assign(n, 0);
            initialize_ac<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(C), ptr(k), ptr(ac));
            if (debug) {
                float acsum = thrust_sum(ac);                             
                assert(abs(acsum - 2 * weights_sum) < 0.001 * acsum);
            }
            
            Qp = Qc;
            initialize_uniqueC_and_C(n, C, uniqueC, c);

            dvf dQc(BLOCKS_NUMBER);
            calculate_modularity<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, c, ptr(V), ptr(N), ptr(W), ptr(C), 
                                                            ptr(uniqueC), ptr(ac), weights_sum, ptr(dQc));
            Qc = thrust_sum(dQc);

            
            std::cerr << "modularity: " << Qc << std::endl;

        } while (abs(Qc - Qp) > threshold);

        //AGGREGATION PHASE
        degree = dvi(n, 0); 
        initialize_degree<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(V), ptr(W), ptr(degree));

        dvi comDegree(n, 0);
        initialize_comm_degree<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(degree), ptr(C), ptr(comDegree));

        comm_size = dvi(n, 0);
        initialize_comm_size<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(C), ptr(comm_size));
    
        dvi newID(n, 0);
        initialize_newID<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(C), ptr(comm_size), ptr(newID));
        thrust_inclusive_scan(newID);
        thrust_sub_for_each(newID, 1);

        dvi edge_pos = comDegree;
        thrust_inclusive_scan(edge_pos);

        dvi vertex_start = comm_size;
        thrust_inclusive_scan(vertex_start);

        dvi comm(n);
        initialize_comm<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(C), ptr(comm), ptr(vertex_start));

        //merge community
        int aggregated_n; 
        int aggregated_m; 
        dvi aggregated_V;
        dvi aggregated_N; 
        dvf aggregated_W;

        aggregated_n = newID.back() + 1;
        aggregated_m = edge_pos.back();

        aggregated_V = dvi(aggregated_n + 1, 0);
        initialize_aggregated_V<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(C), ptr(newID), ptr(edge_pos), ptr(aggregated_V));

        aggregated_N = dvi(aggregated_m, -1);
        aggregated_W = dvf(aggregated_m, NO_EDGE);

        dvi hash_size = dvi(aggregated_n);
        thrust_copy_if_non_zero(comDegree, comm_size, hash_size);   

        dvi hash_offset;
        dvi hash_comm;
        dvf hash_weight;
        thrust_transform_hashmap_size(hash_size, hash_size, 1.5);
        hashmap_create(hash_size, hash_offset, hash_comm, hash_weight);
       
        merge_community_fill_hashmap<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(n, ptr(V), ptr(N),  ptr(W), ptr(C), ptr(comm), 
                                ptr(degree), ptr(newID), ptr(hash_offset), ptr(hash_comm), ptr(hash_weight), debug);
        merge_community_initialize_graph<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(ptr(hash_offset), ptr(hash_comm), 
                                ptr(hash_weight), aggregated_n, ptr(aggregated_V),  ptr(aggregated_N), ptr(aggregated_W));

        save_final_communities<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(initial_n, ptr(finalC), ptr(C), ptr(newID));

        n = aggregated_n; 
        m = aggregated_m; 
        V = aggregated_V;
        N = aggregated_N; 
        W = aggregated_W;
        break;
    } while (abs(Qc - Qba)> threshold);

    std::cout << std::fixed << Qc << std::endl;

    float elapsed_time = stop_recording_time(start_time, stop_time);
    printf("%3.1f ms\n", elapsed_time);

    if (show_assignment) {
        hvi host_finalC = finalC;
        vi stl_finalC(ptr(host_finalC), ptr(host_finalC) + initial_n); 
        print_clustering(initial_n, stl_finalC);
    }
    return 0;
}


// struct Bucket {
//     int from;
//     int to;

//     Bucket(int _from, int _to) : from(_from), to(_to) {};

//     __host__ __device__
//     bool operator()(const int &x) const {
//         return from <= x && x <= to;
//     }
// };

// dvi reorder(n);
// thrust_sequence(reorder);
// degree_copy = degree;
// thrust::partition(thrust::make_zip_iterator(thrust::make_tuple(degree_copy.begin(), reoreder.begin())),
//                     thrust::make_zip_iterator(thrust::make_tuple(degree_copy.end(), reorder.end())),
//                     degree_copy.begin(),
//                     Bucket(3,7));