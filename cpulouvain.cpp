#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>  
#include <vector>
#include <map>


// TODO bigger threshold int the first iteration
// TODO bucket partition
// TODO two hashmaps; hash function

using namespace std;

using pi = pair<int, int>;
using tr = pair<pi, float>;
using vi = vector<int>;
using vf = vector<float>;

float ITR_MODULARITY_THRESHOLD = 0.1;

template<typename T>
void head(vector<T> v, int n = 5) {
    for (int i = 0; i < min(n, (int) v.size()); i++) {
         cout << v[i] << " ";
    }
    cout << endl;
}

template<typename T>
T sum(vector<T> v) {
    int sum = 0;
    for (auto val : v) {
        sum += val;
    }
    return sum;
}

template<typename T>
vector<T> cumsum(vector<T> v) {
    vector<T> cs;
    int sum = 0;
    for (auto val : v) {
        sum += val;
        cs.push_back(sum);
    }
    return cs;
}

void compute_move(int node, vi& newComm) {
    map<int, float> hashMap;

}

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


    vi V(n + 1, 0);
    vi N(2 * m, 0);
    vf W(2 * m, 0);

    int v_idx = 0;
    for (int i = 0; i < tmp.size(); i++) {
        while (v_idx <= tmp[i].first.first) {
            V[v_idx++] = i;
        }
        N[i] = tmp[i].first.second;
        W[i] = tmp[i].second;
    }
    V[v_idx] = 2 * m;


    vi C(n, 0);
    for (int i = 0; i < n; ++i) {
        C[i] = i;
    }

    float wm = sum(W);

    vf k(n, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = V[i]; j < V[i + 1]; ++j) {
            k[i] += W[j];
        }
    }

    vf ac = k;

    float old_modularity = 0;
    float new_modularity = 0;

    vi newComm(n,0);
    int itr = 0;

    while (itr == 0 || new_modularity - old_modularity > ITR_MODULARITY_THRESHOLD) {
        for (int i = 0; i < n; ++i) {
            compute_move(i, newComm);
        }
        C = newComm;
        ac.assign(ac.size(), 0);
        for (int i = 0; i < n; ++i) {
            ac[C[i]] += k[i];
        }


        itr++;
    }

    head(ac);


}