# Requires the mvcc gem (README.md, "Install").
require "mvcc"

w, h = 80, 56
k = Mvcc.compile(<<~'CU')
  #include <cuda_runtime.h>
  #include <math.h>
  __device__ void qmul(float ax, float ay, float az, float aw,
                      float bx, float by, float bz, float bw,
                      float* x, float* y, float* z, float* w) {
    *x = aw * bx + ax * bw + ay * bz - az * by;
    *y = aw * by - ax * bz + ay * bw + az * bx;
    *z = aw * bz + ax * by - ay * bx + az * bw;
    *w = aw * bw - ax * bx - ay * by - az * bz;
  }
  __device__ unsigned char sat(float x) {
    int v = (int)(fminf(fmaxf(x, 0.f), 1.f) * 255.f);
    return (unsigned char)v;
  }
  __device__ float de(float px, float py, float pz,
                     float cx, float cy, float cz, float cw) {
    float zx = px, zy = py, zz = pz, zw = 0.f;
    float dx = 1.f, dy = 0.f, dz = 0.f, dw = 0.f;
    for (int i = 0; i < 9; i++) {
      float nx, ny, nz, nw;
      qmul(zx, zy, zz, zw, dx, dy, dz, dw, &nx, &ny, &nz, &nw);
      dx = 2.f * nx; dy = 2.f * ny; dz = 2.f * nz; dw = 2.f * nw;
      qmul(zx, zy, zz, zw, zx, zy, zz, zw, &nx, &ny, &nz, &nw);
      zx = nx + cx; zy = ny + cy; zz = nz + cz; zw = nw + cw;
      if (zx * zx + zy * zy + zz * zz + zw * zw > 8.f) break;
    }
    float r = sqrtf(zx * zx + zy * zy + zz * zz + zw * zw);
    float dr = sqrtf(dx * dx + dy * dy + dz * dz + dw * dw);
    return 0.5f * r * logf(fmaxf(r, 1e-8f)) / fmaxf(dr, 1e-8f);
  }
  __global__ void trace(unsigned char* rgb, int w, int h, int tick) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h) return;
    float tm = tick * 0.028f;
    float cx = -0.2f + 0.13f * cosf(tm);
    float cy = 0.58f + 0.10f * sinf(tm * 0.77f);
    float cz = 0.21f + 0.10f * sinf(tm * 1.13f);
    float cw = 0.18f + 0.07f * cosf(tm * 0.51f);
    float u = ((i % w) + 0.5f) / w * 2.f - 1.f;
    float v = 1.f - ((i / w) + 0.5f) / h * 2.f;
    u *= (float)w / (float)h * 0.55f;
    float a = tm * 0.35f + 0.7f, ca = cosf(a), sa = sinf(a);
    float rad = 1.85f + 0.15f * sinf(tm * 0.4f);
    float ox = sa * rad, oy = 0.22f, oz = ca * rad;
    float fx = -ox, fy = -oy, fz = -oz;
    float fl = sqrtf(fx * fx + fy * fy + fz * fz);
    fx /= fl; fy /= fl; fz /= fl;
    float rx = fz, rz = -fx, rl = sqrtf(rx * rx + rz * rz);
    rx /= rl; rz /= rl;
    float ux = -rz * fy, uy = rz * fx - rx * fz, uz = rx * fy;
    float dx = fx + u * rx + v * ux, dy = fy + v * uy, dz = fz + u * rz + v * uz;
    float dl = sqrtf(dx * dx + dy * dy + dz * dz);
    dx /= dl; dy /= dl; dz /= dl;
    float t = 0.f;
    int hit = 0;
    for (int s = 0; s < 72; s++) {
      float d = de(ox + t * dx, oy + t * dy, oz + t * dz, cx, cy, cz, cw);
      if (d < 0.001f) { hit = 1; break; }
      t += d;
      if (t > 8.f) break;
    }
    float r = 0.03f + 0.04f * (1.f - v);
    float g = 0.04f + 0.03f * (1.f - v);
    float b = 0.07f + 0.08f * (1.f - v);
    if (hit) {
      float px = ox + t * dx, py = oy + t * dy, pz = oz + t * dz, e = 0.0015f;
      float nx = de(px + e, py, pz, cx, cy, cz, cw) - de(px - e, py, pz, cx, cy, cz, cw);
      float ny = de(px, py + e, pz, cx, cy, cz, cw) - de(px, py - e, pz, cx, cy, cz, cw);
      float nz = de(px, py, pz + e, cx, cy, cz, cw) - de(px, py, pz - e, cx, cy, cz, cw);
      float nl = sqrtf(nx * nx + ny * ny + nz * nz) + 1e-8f;
      nx /= nl; ny /= nl; nz /= nl;
      float lx = 0.45f, ly = 0.75f, lz = 0.48f;
      float ll = sqrtf(lx * lx + ly * ly + lz * lz);
      lx /= ll; ly /= ll; lz /= ll;
      float dif = fmaxf(0.f, nx * lx + ny * ly + nz * lz);
      float fre = powf(1.f - fmaxf(0.f, -(nx * dx + ny * dy + nz * dz)), 3.f);
      float coral = 0.5f + 0.5f * ny;
      r = (0.95f * coral + 0.15f) * (0.12f + 0.88f * dif) + fre;
      g = (0.28f + 0.45f * (1.f - coral)) * (0.12f + 0.88f * dif) + fre * 0.9f;
      b = (0.22f + 0.75f * (1.f - coral)) * (0.12f + 0.88f * dif) + fre * 0.85f;
      float fog = expf(-t * 0.18f);
      r = r * fog + 0.03f * (1.f - fog);
      g = g * fog + 0.04f * (1.f - fog);
      b = b * fog + 0.07f * (1.f - fog);
    }
    rgb[i * 3] = sat(r);
    rgb[i * 3 + 1] = sat(g);
    rgb[i * 3 + 2] = sat(b);
  }
  extern "C" void render(unsigned char* host, int w, int h, int tick) {
    static unsigned char* d; static int cap;
    int n = w * h * 3;
    if (n > cap) { if (d) cudaFree(d); cudaMalloc(&d, n); cap = n; }
    trace<<<(w * h + 255) / 256, 256>>>(d, w, h, tick);
    cudaDeviceSynchronize();
    cudaMemcpy(host, d, n, cudaMemcpyDeviceToHost);
  }
CU

buf = Mvcc.buffer(w * h * 3)
at_exit { print "\e[?25h" } if $stdout.tty?
print "\e[2J\e[?25l" if $stdout.tty?

($stdout.tty? ? 0.step : 1.times).each do |t|
  k.render(buf, w, h, t)
  print "\e[H" if $stdout.tty?
  puts buf.halfblocks(w)
  sleep 0.04
end
