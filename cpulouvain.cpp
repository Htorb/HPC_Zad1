#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>  
#include <vector>
#include <map>


// TODO bigger threshold int the first iteration
// TODO bucket partition
// TODO two hashmaps; hash function
// are weights positive?
// TODO how can update modularity

using namespace std;

using pi = pair<int, int>;
using tr = pair<pi, float>;
using vi = vector<int>;
using vf = vector<float>;

float ITR_MODULARITY_THRESHOLD = 0.1;

// class Graph {
// public:

// }

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

void computeMove(int i,
                 vi& newComm, 
                 const vi& V,
                 const vi& N, 
                 const vf& W, 
                 const vi& C, 
                 const vf& k, 
                 const vf& ac, 
                 const float wm) {
    map<int, float> hashMap;
    int ci = C[i];

    hashMap[ci] = 0;

    for (int j = V[i]; j < V[i + 1]; ++j) {
        int cj = C[N[j]];
        if (hashMap.count(cj) == 0) {
            hashMap[cj] = 0;
        }
        if (N[j] != i) {
            hashMap[cj] += W[j];
        }
    }

    int maxCj = -1;
    float maxDeltaAlmostMod = -1;

    for (auto const& [cj, wsum] : hashMap) {
        if (cj < ci) {
            float deltaAlmostMod = wsum / wm 
                + k[i] * (ac[ci] - k[i] - ac[cj]) / (2 * wm * wm);

            if (deltaAlmostMod > maxDeltaAlmostMod || deltaAlmostMod == maxDeltaAlmostMod && cj < maxCj) {
                maxCj = cj;
                maxDeltaAlmostMod = deltaAlmostMod;
            }   
        }
    }

    float maxDeltaMod = maxDeltaAlmostMod - hashMap[ci] / wm;
    if (maxDeltaMod > 0) {
        newComm[i] = maxCj;
    } else {
        newComm[i] = ci;
    }
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
        if (v1 != v2) {
            tmp.push_back(tr(pi(v2 - 1,v1 - 1),f));
        }
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
    while (itr < 8 || new_modularity - old_modularity > ITR_MODULARITY_THRESHOLD) {
        for (int i = 0; i < n; ++i) {
            computeMove(i, newComm, V, N, W, C, k, ac, wm);
        }
        C = newComm;
        ac.assign(ac.size(), 0);
        for (int i = 0; i < n; ++i) {
            ac[C[i]] += k[i];
        }
        itr++;

        //modularity calculation
        float Q = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = V[i]; j < V[i + 1]; ++j) {
                if (C[N[j]] == C[i]) {
                    Q += W[j] / (2 * wm);
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            Q -= ac[i] * ac[i] / (4 * wm * wm);
        }

        cout << "modularity: " << Q << endl;
    }

    head(newComm);


}