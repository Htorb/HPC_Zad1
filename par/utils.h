#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

using vi = std::vector<int>;
using vf = std::vector<float>;
using pi = std::pair<int, int>;
using tr = std::pair<pi, float>;

void parse_command_line(bool&        showAssignment,
                      float&       threshold,
                      std::string& matrixFile,
                      int          argc,
                      char**       argv,
                      bool&        DEBUG);

void print_clustering(int        initialN, 
                     vi&  finalC);                    

void read_graph_from_file(std::string& matrixFile, 
                       int&         n,
                       int&         m,
                       vi&          V, 
                       vi&          N,
                       vf&          W);

void start_recording_time(cudaEvent_t& startTime,
                        cudaEvent_t& stopTime);

float stop_recording_time(cudaEvent_t& startTime,
                        cudaEvent_t& stopTime);

#endif