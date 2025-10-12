#ifndef OPENMP_HPP
#define OPENMP_HPP


#if defined(_OPENMP)
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_num_threads() { return 1; }
inline void omp_set_num_threads(int) {}
#endif

#endif // OPENMP_HPP