#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>  
#include <vector>


using namespace std;

using pi = pair<int, int>;
using tr = pair<pi, float>;

float ITR_MODULARITY_THRESHOLD = 0.1;

int main(int argc, char *argv[]) {
    bool show_assignment = false;
    float threshold = 0;
    string matrix_file;

    int i = 1;
    while (i < argc) {
        string s(argv[i]);
        if (s == "-f") {
            matrix_file = string(argv[i + 1]);
            i += 2;
        } else if (s == "-g") {
            threshold = strtof(argv[i + 1], NULL);
            i += 2;
        } else if (s == "-v") {
            show_assignment = true;
            i += 1;
        } else {
            exit(1);
        }
    }


   
    ifstream matrix_stream;
    matrix_stream.open(matrix_file);
    
    int n, m;
    matrix_stream >> n >> n >> m;
    

    vector<tr> tmp;

    for (int i = 0; i < m; i++) {
        int v1, v2;
        float f;
        matrix_stream >> v1 >> v2 >> f;
        tmp.push_back(tr(pi(v1 - 1,v2 - 1),f));
        tmp.push_back(tr(pi(v2 - 1,v1 - 1),f));
    }

    sort(tmp.begin(), tmp.end());


    vector<int> V(n + 1, 0);
    vector<int> N(2 * m, 0);
    vector<float> W(2 * m, 0);

    int v_idx = 0;
    for (int i = 0; i < tmp.size(); i++) {
        while (v_idx <= tmp[i].first.first) {
            V[v_idx++] = i;
        }
        N[i] = tmp[i].first.second;
        W[i] = tmp[i].second;
    }
    V[v_idx] = 2 * m;

    // for (int i = 0; i < V.size(); i++) {
    //     cout << V[i] << endl;
    // }

    float old_modularity = 0;
    float new_modularity = 0;

    int itr = 0;
    while (itr == 0 || new_modularity - old_modularity > ITR_MODULARITY_THRESHOLD) {

        itr++;
    }


}