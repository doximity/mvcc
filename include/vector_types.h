// mvcc: CUDA built-in vector types (public CUDA C++ Programming Guide, "Built-in Vector Types").
#ifndef MVCC_VECTOR_TYPES_H
#define MVCC_VECTOR_TYPES_H
#include <mvcc/host_defines.h>

#define MVCC_VEC1(N, T) struct N { T x; };
#define MVCC_VEC2(N, T, A) struct __align__(A) N { T x, y; };
#define MVCC_VEC3(N, T) struct N { T x, y, z; };
#define MVCC_VEC4(N, T, A) struct __align__(A) N { T x, y, z, w; };

MVCC_VEC1(char1, signed char) MVCC_VEC2(char2, signed char, 2) MVCC_VEC3(char3, signed char) MVCC_VEC4(char4, signed char, 4)
MVCC_VEC1(uchar1, unsigned char) MVCC_VEC2(uchar2, unsigned char, 2) MVCC_VEC3(uchar3, unsigned char) MVCC_VEC4(uchar4, unsigned char, 4)
MVCC_VEC1(short1, short) MVCC_VEC2(short2, short, 4) MVCC_VEC3(short3, short) MVCC_VEC4(short4, short, 8)
MVCC_VEC1(ushort1, unsigned short) MVCC_VEC2(ushort2, unsigned short, 4) MVCC_VEC3(ushort3, unsigned short) MVCC_VEC4(ushort4, unsigned short, 8)
MVCC_VEC1(int1, int) MVCC_VEC2(int2, int, 8) MVCC_VEC3(int3, int) MVCC_VEC4(int4, int, 16)
MVCC_VEC1(uint1, unsigned int) MVCC_VEC2(uint2, unsigned int, 8) MVCC_VEC3(uint3, unsigned int) MVCC_VEC4(uint4, unsigned int, 16)
MVCC_VEC1(long1, long) MVCC_VEC2(long2, long, 16) MVCC_VEC3(long3, long) MVCC_VEC4(long4, long, 16)
MVCC_VEC1(ulong1, unsigned long) MVCC_VEC2(ulong2, unsigned long, 16) MVCC_VEC3(ulong3, unsigned long) MVCC_VEC4(ulong4, unsigned long, 16)
MVCC_VEC1(longlong1, long long) MVCC_VEC2(longlong2, long long, 16) MVCC_VEC3(longlong3, long long) MVCC_VEC4(longlong4, long long, 16)
MVCC_VEC1(ulonglong1, unsigned long long) MVCC_VEC2(ulonglong2, unsigned long long, 16) MVCC_VEC3(ulonglong3, unsigned long long) MVCC_VEC4(ulonglong4, unsigned long long, 16)
MVCC_VEC1(float1, float) MVCC_VEC2(float2, float, 8) MVCC_VEC3(float3, float) MVCC_VEC4(float4, float, 16)
MVCC_VEC1(double1, double) MVCC_VEC2(double2, double, 16) MVCC_VEC3(double3, double) MVCC_VEC4(double4, double, 16)

#undef MVCC_VEC1
#undef MVCC_VEC2
#undef MVCC_VEC3
#undef MVCC_VEC4

struct dim3 {
  unsigned int x, y, z;
#if defined(__cplusplus)
  __host__ __device__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
  __host__ __device__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
  __host__ __device__ constexpr operator uint3() const { return uint3{x, y, z}; }
#endif
};

#endif // MVCC_VECTOR_TYPES_H
