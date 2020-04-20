#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>  
#include <vector>
#include <map>
#include <set>


// TODO bigger threshold int the first iteration
// TODO bucket partition
// TODO two hashmaps; hash function
// are weights positive?
// TODO how can update modularity
// TODO is the graph directed?

using namespace std;

using pi = pair<int, int>;
using tr = pair<pi, float>;
using vi = vector<int>;
using vf = vector<float>;

//float ITR_MODULARITY_THRESHOLD = 0.1;
float NO_EDGE = -1;
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
void printVec(vector<T> v){
    head(v, v.size());
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
void cumsum(vector<T>& v) {
    int sum = 0;
    for (int i = 0; i < v.size(); ++i) {
        sum += v[i];
        v[i] = sum;
    }
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
        if (N[j] == NO_EDGE)
            break;
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

            if (deltaAlmostMod > maxDeltaAlmostMod || deltaAlmostMod == maxDeltaAlmostMod && cj < maxCj) { //TODO change to correct
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


float calculateModularity(const vi& V,
                            const vi& N, 
                            const vf& W, 
                            const vi& C, 
                            const vf& ac, 
                            const float wm) {
    int n = ac.size();
    float Q = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = V[i]; j < V[i + 1]; ++j) {
            if (N[j] == NO_EDGE)
                break; 
            if (C[N[j]] == C[i]) {
                Q += W[j] / (2 * wm);
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        Q -= ac[i] * ac[i] / (4 * wm * wm);
    }
    return Q;
}


int main(int argc, char *argv[]) {
    bool showAssignment = false;
    float threshold = 0;
    string matrixFile;

    int n; //number vertices 
    int m; //number of edges
    vi V; //vertices
    vi N; //neighbours
    vf W; //weights
    float wm; //sum of weights
    vi C; //current clustering
    vf k; //sum of vertex's edges
    vf ac; //sum of cluster edges

    vi finalC; //final clustering result 

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
        } else {
            exit(1);
        }
    }

   
    ifstream matrixStream;
    matrixStream.open(matrixFile);
    int entries = 0;
    matrixStream >> n >> n >> entries;
    
    m = 0;
    wm = 0;
    vector<tr> tmp;
    for (int i = 0; i < entries; i++) {
        int v1, v2;
        float f;
        matrixStream >> v1 >> v2 >> f;

        m++;
        wm += f;
        tmp.push_back(tr(pi(v1 - 1,v2 - 1),f));
        //if graph is undirected
        if (v1 != v2) {
            m++;
            tmp.push_back(tr(pi(v2 - 1,v1 - 1),f));
        }
    }

    sort(tmp.begin(), tmp.end());


    V = vi(n + 1, 0);
    N = vi(m, 0);
    W = vf(m, 0);

    int v_idx = 0;
    for (int i = 0; i < tmp.size(); i++) {
        while (v_idx <= tmp[i].first.first) {
            V[v_idx++] = i;
        }
        N[i] = tmp[i].first.second;
        W[i] = tmp[i].second;
    }
    while (v_idx < n + 1) {
        V[v_idx++] = m;
    }

    C = vi(n, 0);
    for (int i = 0; i < n; ++i) {
        C[i] = i;
    }

    k = vf(n, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = V[i]; j < V[i + 1]; ++j) {
            k[i] += W[j];
        }
    }

    ac = k;

    finalC = C;
    int itr = 0;
    while (itr < 2) {

        float oldQ;
        float Q = calculateModularity(V, N, W, C, ac, wm);
        do {
            cout << "modularity: " << Q << endl;
            cout << "clasters: ";
            head(C, C.size());


            vi newComm = C;
            for (int i = 0; i < n; ++i) {
                computeMove(i, newComm, V, N, W, C, k, ac, wm);
            }
            C = newComm;
            ac.assign(ac.size(), 0);
            for (int i = 0; i < n; ++i) {
                ac[C[i]] += k[i];
            }

            oldQ = Q;
            Q = calculateModularity(V, N, W, C, ac, wm);

        } while (Q - oldQ > threshold);

        //aggregation phase
        vi comSize(n, 0);
        vi comDegree(n, 0);



        for (int i = 0; i < n; ++i) {
            comSize[C[i]] += 1;
            comDegree[C[i]] += V[i + 1] - V[i]; //works but only in the first iteration
        }



        vi newID(n, 0);
        for (int i = 0; i < n; ++i) {
            if (comSize[C[i]] != 0) {
                newID[C[i]] = 1;
            }
        }
        

        cumsum(newID);

        vi edgePos = comDegree;
        cumsum(edgePos);

        vi vertexStart = comSize;
        cumsum(vertexStart);

        vi comm(n, 0);
        for (int i = 0; i < n; ++i) {
            vertexStart[C[i]] -= 1; //in paper is add
            int res = vertexStart[C[i]];
            comm[res] = i; 
        }

         
        //merge community
        //new graph
        int newn; 
        int newm; 
        vi newV;
        vi newN; 
        vf newW;
        vi newC; 
        vf newk; 
        vf newac; 

        newn = newID.back();
        newm = edgePos.back();
        newV = vi(newn + 1, 0);
        for (int i = 0; i < n; ++i) {
            newV[newID[C[i]]] = edgePos[C[i]];
        }

        newN = vi(newm, NO_EDGE);
        newW = vf(newm, 0);


        map<int, float> hashMap;
        int oldc = C[comm[0]]; //can be n = 0?
        for (int idx = 0; idx <= n; ++idx) {
            if (idx == n || oldc != C[comm[idx]]) {
                int edgeId = newV[newID[oldc] - 1];


                for (auto const& [cj, wsum] : hashMap) {

                    newN[edgeId] = newID[cj] - 1;
                    newW[edgeId] = wsum;
                    edgeId++;
                }

                if (idx != n) {
                    oldc = C[comm[idx]];
                }

                hashMap.clear();
            } else {
                int i = comm[idx];
                int ci = C[i];
                for (int j = V[i]; j < V[i + 1]; ++j) {
                    if (N[j] == NO_EDGE)
                        break; 
                    int cj = C[N[j]];
                    if (hashMap.count(cj) == 0) {
                        hashMap[cj] = 0;
                    }
                    if (cj != ci || i <= N[j])
                        hashMap[cj] += W[j];
                }
            }
        }

        newC = vi(newn, 0);
        for (int i = 0; i < newn; ++i) {
            newC[i] = i;
        }

        newk = vf(newn, 0);
        for (int i = 0; i < newn; ++i) {
            for (int j = newV[i]; j < newV[i + 1]; ++j) {
                if (N[j] == NO_EDGE)
                    break;
                newk[i] += newW[j];
            }
        }

        newac = newk;

        for (int i = 0; i < finalC.size(); ++i) {
            finalC[i] = newID[C[finalC[i]]] - 1;
        }

        //update graph
        n = newn; 
        m = newm; 
        V = newV;
        N = newN; 
        W = newW;
        C = newC; 
        k = newk; 
        ac = newac;

        

        itr++;
    }


    if (showAssignment) {
        vector<pi> finalCPrime;
        
        for (int i = 0; i < finalC.size(); ++i) {
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

    



    return 0;
}