#include "thrust_wrappers.h"

using namespace thrust::placeholders;

struct is_non_zero {
    __host__ __device__
    bool operator()(const int &x) const {
      return x != 0;
    }
};

struct hash_map_size_generator {
    const float ratio;
    hash_map_size_generator(float _ratio) : ratio(_ratio) {}
    __host__ __device__ 
   int operator()(const int &x) const {
        if (x == 0) return 0;
        int v = (int) (ratio * (x + 1));
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }
};


void thrust_sort(dvi& v) {
    thrust::sort(v.begin(), v.end());
}

int thrust_unique(dvi& v) { 
    auto end = thrust::unique(v.begin(), v.end());
    return end - v.begin();
}

float thrust_sum(dvf& v) {
    return thrust::reduce(v.begin(), v.end(), (float) 0, thrust::plus<float>());
}

void thrust_sequence(dvi& v) {
    thrust::sequence(v.begin(), v.end()); 
}

void thrust_inclusive_scan(dvi& v) {
    thrust::inclusive_scan(v.begin(), v.end(), v.begin());
}

void thrust_inclusive_scan_with_shift(dvi& from, dvi& to, int val) {
    thrust::inclusive_scan(from.begin(), from.end(), to.begin() + val);
}

void thrust_copy_if_non_zero(dvi& from, dvi& iff, dvi& to) {
    thrust::copy_if(from.begin(), from.end(), iff.begin(), to.begin(), is_non_zero());
}

void thrust_transform_hashmap_size(dvi& from, dvi& to, float frac) {
    thrust::transform(from.begin(), from.end(), to.begin(), hash_map_size_generator(frac));
}

void thrust_sub_for_each(dvi& v, int val) {
    thrust::for_each(v.begin(), v.end(), _1 -= val);
}

void thrust_sort_graph(dvi& V, dvi& N, dvf& W) {
    thrust::sort_by_key(V.begin(), V.end(), thrust::make_zip_iterator(make_tuple( V.begin(), N.begin(), W.begin())));
}

void thrust_reduce_by_key(dvi& A, dvi& B, dvi& C, dvi& D) {
    thrust::reduce_by_key(A.begin(), A.end(), B.begin(), C.begin(), D.begin());
}