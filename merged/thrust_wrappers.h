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
