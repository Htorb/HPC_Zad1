#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>  
#include <vector>
#include <map>
#include <set>
#include <cassert>  


// TODO bigger threshold int the first iteration
// TODO bucket partition
// TODO two hashmaps; hash function
// TODO how can update modularity
#define dassert(a) \
    if ((debug)) assert((a));

#define pvec(a) \
    do { std::cout << #a << ": "; printVec((a)) ; } while(false)

using namespace std;

using pi = pair<int, int>;
using tr = pair<pi, float>;
using vi = vector<int>;
using vf = vector<float>;
using vb = vector<bool>;


//float ITR_MODULARITY_THRESHOLD = 0.1;
float NO_EDGE = -1;

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
T positiveSum(vector<T> v) {
    int sum = 0;
    for (auto val : v) {
        if (val > 0) {
            sum += val;
        }
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

void parseCommandline(bool& showAssignment,
                        float& threshold,
                        string& matrixFile,
                        bool& debug,
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
            debug = true;
            i += 1;
        } else {
            exit(1);
        }
    }
}

void readGraphFromFile(const string& matrixFile, 
                        int& n,
                        int& m,
                        vi& V, 
                        vi& N,
                        vf& W) {
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
}


void computeMove(int i,
                    vi& newComm, 
                    const vi& V,
                    const vi& N, 
                    const vf& W, 
                    const vi& C,
                    const vb& isCommunityByItself, 
                    const vf& k, 
                    const vf& ac, 
                    const float wm) {
    map<int, float> hashMap;
    int ci = C[i];

    hashMap[ci] = 0;
    for (int j = V[i]; j < V[i + 1]; ++j) {
        if (W[j] == NO_EDGE)
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

            if (deltaAlmostMod > maxDeltaAlmostMod || deltaAlmostMod == maxDeltaAlmostMod && cj < maxCj) {
                if (!isCommunityByItself[cj] || !isCommunityByItself[ci] || cj < ci) {
                    maxCj = cj;
                    maxDeltaAlmostMod = deltaAlmostMod;
                }
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
            if (W[j] == NO_EDGE)
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

void initializeCommunities(int n, vi& C) {
    for (int i = 0; i < n; ++i) {
            C[i] = i;
    }
}

void initializeK(int n, const vi& V, const vf& W, vf& k) {
    for (int i = 0; i < n; ++i) {
        for (int j = V[i]; j < V[i + 1]; ++j) {
            if (W[j] == NO_EDGE)
                break;
            k[i] += W[j];
        }
    }
}

void initializeAc(int n, const vi& C, const vf& k, vf& ac) {
    for (int i = 0; i < n; ++i) {
        ac[C[i]] += k[i];
    }
}

void initializeIsCommunityByItself(int n, const vi& C, vb& isCommunityByItself) {
    for (int i = 0; i < n; ++i) {
        isCommunityByItself[C[i]] = true;
    }
}

int main(int argc, char *argv[]) {
    //commandline vars
    bool showAssignment = false;
    float threshold = 0;
    string matrixFile;
    bool debug = false;

    //graph vars
    int n; //number vertices 
    int m; //number of edges
    vi V; //vertices
    vi N; //neighbours
    vf W; //weights
    float wm; //sum of weights
    vi C; //current clustering
    vf k; //sum of vertex's edges
    vf ac; //sum of cluster edges
    vb isCommunityByItself; 


    vi finalC; //final clustering result 

   
    parseCommandline(showAssignment, threshold, matrixFile, debug, argc, argv);
    readGraphFromFile(matrixFile, n, m, V, N, W);
    wm = sum(W) / 2;

    C = vi(n, 0);
    initializeCommunities(n, C);

    k = vf(n, 0);
    initializeK(n, V, W, k);

    ac = k; 
    finalC = C;

    vi newComm;

    int itr = 0;
    while (itr < 2) { //TODO change loop condition
        float Qp;
        float Qc = calculateModularity(V, N, W, C, ac, wm);
        cout << wm << endl;

        if (debug) {
            cout << "modularity: " << Qc << endl;
            pvec(C);
        }
        do {
            newComm = C;
            isCommunityByItself = vb(n, false);
            initializeIsCommunityByItself(n, C, isCommunityByItself);

            for (int i = 0; i < n; ++i) {
                computeMove(i, newComm, V, N, W, C, isCommunityByItself, k, ac, wm);
            }

            C = newComm;
            ac.assign(n, 0);
            initializeAc(n, C, k, ac);
            
            dassert(abs(sum(ac) - 2 * wm) < 0.0001);

            Qp = Qc;
            Qc = calculateModularity(V, N, W, C, ac, wm);

            if (debug) {
                cout << "modularity: " << Qc << endl;
                pvec(C);
            }

        } while (Qc - Qp > threshold);

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

        newN = vi(newm, -1);
        newW = vf(newm, NO_EDGE);


        map<int, float> hashMap;
        int oldc = C[comm[0]]; //can be n = 0?
        for (int idx = 0; idx < n; ++idx) {
            int i = comm[idx];
            int ci = C[i];
            if (oldc != ci) {
                int edgeId = newV[newID[oldc] - 1];
                for (auto const& [cj, wsum] : hashMap) {
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
                //cout << "edge from " << comm[idx] << " to " << N[j] << " with weight " << W[j] << " aggregated to cluster: " << C[comm[idx]] << endl;
                hashMap[cj] += W[j];
            }
            
        }
        int edgeId = newV[newID[oldc] - 1];
        for (auto const& [cj, wsum] : hashMap) {
            newN[edgeId] = newID[cj] - 1;
            newW[edgeId] = wsum;
            edgeId++;
        }


        newC = vi(newn, 0);
        for (int i = 0; i < newn; ++i) {
            newC[i] = i;
        }

        newk = vf(newn, 0);
        for (int i = 0; i < newn; ++i) {
            for (int j = newV[i]; j < newV[i + 1]; ++j) {
                if (newW[j] == NO_EDGE)
                    break;
                newk[i] += newW[j];
            }
        }

        newac = newk;

        for (int i = 0; i < finalC.size(); ++i) {
            finalC[i] = newID[C[finalC[i]]] - 1;
        }
        
        if (debug) {
            cout << "sum of weights:" << positiveSum(W) << endl;
            pvec(W);
            pvec(N);
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