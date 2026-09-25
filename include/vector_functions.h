// mvcc: make_<type> constructors for the built-in vector types.
#ifndef MVCC_VECTOR_FUNCTIONS_H
#define MVCC_VECTOR_FUNCTIONS_H
#include <vector_types.h>

#define MVCC_MK1(N, T) static __inline__ __host__ __device__ N make_##N(T x) { N r; r.x = x; return r; }
#define MVCC_MK2(N, T) static __inline__ __host__ __device__ N make_##N(T x, T y) { N r; r.x = x; r.y = y; return r; }
#define MVCC_MK3(N, T) static __inline__ __host__ __device__ N make_##N(T x, T y, T z) { N r; r.x = x; r.y = y; r.z = z; return r; }
#define MVCC_MK4(N, T) static __inline__ __host__ __device__ N make_##N(T x, T y, T z, T w) { N r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }
#define MVCC_MKALL(B, T) MVCC_MK1(B##1, T) MVCC_MK2(B##2, T) MVCC_MK3(B##3, T) MVCC_MK4(B##4, T)

MVCC_MKALL(char, signed char) MVCC_MKALL(uchar, unsigned char) MVCC_MKALL(short, short) MVCC_MKALL(ushort, unsigned short)
MVCC_MKALL(int, int) MVCC_MKALL(uint, unsigned int) MVCC_MKALL(long, long) MVCC_MKALL(ulong, unsigned long)
MVCC_MKALL(longlong, long long) MVCC_MKALL(ulonglong, unsigned long long) MVCC_MKALL(float, float) MVCC_MKALL(double, double)

#undef MVCC_MK1
#undef MVCC_MK2
#undef MVCC_MK3
#undef MVCC_MK4
#undef MVCC_MKALL
#endif // MVCC_VECTOR_FUNCTIONS_H
