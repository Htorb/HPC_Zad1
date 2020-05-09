#ifndef THRUST_WRAPPERS_H
#define THRUST_WRAPPERS_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

using hvi = thrust::host_vector<int>;
using hvf = thrust::host_vector<float>;
using dvi = thrust::device_vector<int>;
using dvf = thrust::device_vector<float>;

void thrust_sort(dvi&);
int thrust_unique(dvi&);
float thrust_sum(dvf&);
void thrust_sequence(dvi&);
void thrust_inclusive_scan(dvi&);
void thrust_inclusive_scan_with_shift(dvi&, dvi&, int);
void thrust_copy_if_non_zero(dvi&, dvi&, dvi&);
void thrust_transform_hashmap_size(dvi&, dvi&, float);
void thrust_sub_for_each(dvi&, int);

#endif
   
    