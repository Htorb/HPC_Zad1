#ifndef HEADERFILE_H
#define HEADERFILE_H

#include <vector>
#include <string>

using vi = std::vector<int>;
using vf = std::vector<float>;
using pi = std::pair<int, int>;
using tr = std::pair<pi, float>;

void parseCommandline(bool&        showAssignment,
                      float&       threshold,
                      std::string& matrixFile,
                      int          argc,
                      char**       argv,
                      bool&        DEBUG);

void printClustering(int        initialN, 
                     const vi&  finalC);                    

void readGraphFromFile(std::string& matrixFile, 
                       int&         n,
                       int&         m,
                       vi&          V, 
                       vi&          N,
                       vf&          W);

#endif