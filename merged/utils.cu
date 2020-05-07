#include <vector>
#include <iostream>
#include <set>
#include <algorithm>  
#include <fstream>

#include "utils.h"
#include "helpers.h"




void parseCommandline(bool& showAssignment,
                        float& threshold,
                        std::string& matrixFile,
                        int argc,
                        char** argv,
                        bool& DEBUG) {
    int i = 1;
    while (i < argc) {
        std::string s(argv[i]);
        if (s == "-f") {
            matrixFile = std::string(argv[i + 1]);
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

void printClustering(int initialN, vi& finalC) {
    std::vector<pi> finalCPrime;
    for (int i = 0; i < initialN; ++i) {
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

void readGraphFromFile( std::string& matrixFile, 
                        int& n,
                        int& m,
                        vi& V, 
                        vi& N,
                        vf& W) {
    std::ifstream matrixStream;
    matrixStream.open(matrixFile);
    int entries = 0;
    matrixStream >> n >> n >> entries;
    
    m = 0;
    std::vector<tr> tmp;
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

    std::sort(tmp.begin(), tmp.end());

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

void startRecordingTime(cudaEvent_t& startTime, cudaEvent_t& stopTime) {
    HANDLE_ERROR(cudaEventCreate(&startTime));
    HANDLE_ERROR(cudaEventCreate(&stopTime));
    HANDLE_ERROR(cudaEventRecord(startTime, 0));
}

float stopRecordingTime(cudaEvent_t& startTime, cudaEvent_t& stopTime) {
    float elapsedTime;
    HANDLE_ERROR(cudaEventRecord(stopTime, 0));
    HANDLE_ERROR(cudaEventSynchronize(stopTime));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, startTime, stopTime));
    HANDLE_ERROR(cudaEventDestroy(startTime));
    HANDLE_ERROR(cudaEventDestroy(stopTime));
    return elapsedTime;
}
