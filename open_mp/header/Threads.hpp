#ifndef THREADS_HPP
#define THREADS_HPP

#include "OpenMP.hpp"
inline void configure_threads(int nThreads, int chunk = 0) {
#if defined(_OPENMP)
    if (nThreads > 0) omp_set_num_threads(nThreads);
    // nonaktifkan tim dinamis untuk mendapatkan perilaku yang stabil
    omp_set_dynamic(0);
    if (chunk > 0) omp_set_schedule(omp_sched_static, chunk);
#else
    (void)nThreads; (void)chunk;
#endif
}

inline int max_threads() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

#endif 