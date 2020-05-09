#include <vector>
#include <iostream>
#include <set>
#include <algorithm>  
#include <fstream>
#include <stdlib.h>
#include <sstream>

#include "utils.h"
#include "helpers.h"



void parse_command_line(bool& show_assignment,
                        float& threshold,
                        std::string& matrix_file,
                        int argc,
                        char** argv,
                        bool& DEBUG) {
    int i = 1;
    while (i < argc) {
        std::string s(argv[i]);
        if (s == "-f") {
            matrix_file = std::string(argv[i + 1]);
            i += 2;
        } else if (s == "-g") {
            threshold = strtof(argv[i + 1], NULL);
            i += 2;
        } else if (s == "-v") {
            show_assignment = true;
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

void print_clustering(int initial_n, vi& finalC) {
    std::vector<pi> finalCPrime;
    for (int i = 0; i < initial_n; ++i) {
        finalCPrime.push_back(pi(finalC[i], i));
    }
    
    std::cout << std::set<int>(finalC.begin(), finalC.end()).size();
    std::sort(finalCPrime.begin(), finalCPrime.end());
    
    int lastC = -1;
    for (auto& p : finalCPrime) {
        if (lastC != p.first) {
            std::cout << std::endl << p.first;
            lastC = p.first;
        }
        std::cout << " " << p.second;
    }
    std::cout << std::endl;
}

void read_graph_from_file( std::string& matrix_file, 
                        int& n,
                        int& m,
                        vi& V, 
                        vi& N,
                        vf& W) {    
    std::ifstream matrix_stream;
    matrix_stream.open(matrix_file);
    int entries = 0;
    
    std::string line;
    while (std::getline(matrix_stream, line)) {
        if (line[0] != '%') {
            std::stringstream(line) >> n >> n >> entries;
            break;
        }
    }    
    m = 0;
    std::vector<tr> tmp;
    for (int i = 0; i < entries; i++) {
        int v1, v2;
        float f;
        matrix_stream >> v1 >> v2 >> f;

        m++;
        tmp.push_back(tr(pi(v1 - 1,v2 - 1),f));
        if (v1 != v2) {
            m++;
            tmp.push_back(tr(pi(v2 - 1,v1 - 1),f));
        } 
    }
    
    //todo check if sorting is fast
    std::cerr << "starting sort" << std::endl;
    std::sort(tmp.begin(), tmp.end());
    std::cerr << "finished sort" << std::endl;

    V = vi(n + 1, 0);
    N = vi(m, 0);
    W = vf(m, 0);

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

void start_recording_time(cudaEvent_t& start_time, cudaEvent_t& stop_time) {
    HANDLE_ERROR(cudaEventCreate(&start_time));
    HANDLE_ERROR(cudaEventCreate(&stop_time));
    HANDLE_ERROR(cudaEventRecord(start_time, 0));
}

float stop_recording_time(cudaEvent_t& start_time, cudaEvent_t& stop_time) {
    float elapsed_time;
    HANDLE_ERROR(cudaEventRecord(stop_time, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop_time));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start_time, stop_time));
    HANDLE_ERROR(cudaEventDestroy(start_time));
    HANDLE_ERROR(cudaEventDestroy(stop_time));
    return elapsed_time;
}
