# Requires the mvcc gem (README.md, "Install").
require "mvcc"

w, h = 80, 28
k = Mvcc.compile(<<~'CU')
  #include <cuda_runtime.h>
  #include <math.h>
  __device__ void rot(float* a, float* b, float ang) {
    float c = cosf(ang), s = sinf(ang), t = *a * c + *b * s;
    *b = *b * c - *a * s;
    *a = t;
  }
  __device__ float gmod(float v, float m) {
    return v - m * floorf(v / m);
  }
  __device__ float box(float x, float y, float z, float b) {
    x = fabsf(x) - b; y = fabsf(y) - b; z = fabsf(z) - b;
    float m = fmaxf(x, fmaxf(y, z));
    x = fmaxf(x, 0.f); y = fmaxf(y, 0.f); z = fmaxf(z, 0.f);
    return fminf(m, 0.f) + sqrtf(x * x + y * y + z * z);
  }
  /* kaleidoscopic IFS — 6-level twisted Menger lattice */
  __device__ float menger(float x, float y, float z, int depth) {
    float d = box(x, y, z, 1.f);
    float s = 1.f;
    for (int i = 0; i < depth; i++) {
      rot(&x, &y, 0.15f + i * 0.04f);
      s *= 3.f;
      float cx = gmod(x * s, 2.f) - 1.f;
      float cy = gmod(y * s, 2.f) - 1.f;
      float cz = gmod(z * s, 2.f) - 1.f;
      float rx = fabsf(1.f - 3.f * fabsf(cx));
      float ry = fabsf(1.f - 3.f * fabsf(cy));
      float rz = fabsf(1.f - 3.f * fabsf(cz));
      float c = (fminf(fmaxf(rx, ry), fminf(fmaxf(ry, rz), fmaxf(rz, rx))) - 1.f) / s;
      d = fmaxf(d, c);
    }
    return d;
  }
  __device__ float de(float x, float y, float z, float tm) {
    float x1 = x, y1 = y, z1 = z;
    rot(&x1, &z1, tm * 0.31f);
    rot(&y1, &z1, 0.4f + 0.12f * sinf(tm * 0.7f));
    float d = menger(x1, y1, z1, 6);
    float x2 = x * 1.15f, y2 = y * 1.15f, z2 = z * 1.15f;
    rot(&y2, &x2, 1.1f - tm * 0.22f);
    rot(&z2, &x2, 0.85f);
    d = fminf(d, menger(x2, y2, z2, 5) - 0.01f);
    float hole = sqrtf(x * x + y * y + z * z) - 0.42f;
    return fmaxf(d, -hole);
  }
  __device__ void nrm(float x, float y, float z, float tm, float* nx, float* ny, float* nz) {
    float e = 0.0012f;
    *nx = de(x + e, y, z, tm) - de(x - e, y, z, tm);
    *ny = de(x, y + e, z, tm) - de(x, y - e, z, tm);
    *nz = de(x, y, z + e, tm) - de(x, y, z - e, tm);
    float l = sqrtf(*nx * *nx + *ny * *ny + *nz * *nz) + 1e-8f;
    *nx /= l; *ny /= l; *nz /= l;
  }
  __global__ void trace(char* out, int w, int h, int tick) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h) return;
    float tm = tick * 0.04f;
    float u = ((i % w) + 0.5f) / w * 2.f - 1.f;
    float v = 1.f - ((i / w) + 0.5f) / h * 2.f;
    u *= (float)w / (float)h * 0.62f;
    float a = tm * 0.9f + 0.6f, ca = cosf(a), sa = sinf(a);
    float rad = 2.15f + 0.18f * cosf(tm * 0.45f);
    float ox = sa * rad, oy = 0.32f + 0.14f * sinf(tm * 0.6f), oz = ca * rad;
    float fx = -ox, fy = -oy, fz = -oz;
    float fl = sqrtf(fx * fx + fy * fy + fz * fz);
    fx /= fl; fy /= fl; fz /= fl;
    float rx = fz, rz = -fx;
    float rl = sqrtf(rx * rx + rz * rz);
    rx /= rl; rz /= rl;
    float ux = -rz * fy, uy = rz * fx - rx * fz, uz = rx * fy;
    float dx = fx + u * rx + v * ux;
    float dy = fy + v * uy;
    float dz = fz + u * rz + v * uz;
    float dl = sqrtf(dx * dx + dy * dy + dz * dz);
    dx /= dl; dy /= dl; dz /= dl;
    float t = 0.f;
    int hit = 0;
    for (int s = 0; s < 90; s++) {
      float d = de(ox + t * dx, oy + t * dy, oz + t * dz, tm);
      if (d < 0.001f) { hit = 1; break; }
      t += d;
      if (t > 14.f) break;
    }
    if (!hit) { out[i] = ' '; return; }
    float px = ox + t * dx, py = oy + t * dy, pz = oz + t * dz;
    float nx, ny, nz;
    nrm(px, py, pz, tm, &nx, &ny, &nz);
    float lx = 0.4f, ly = 0.8f, lz = 0.45f;
    float ll = sqrtf(lx * lx + ly * ly + lz * lz);
    lx /= ll; ly /= ll; lz /= ll;
    float dif = fmaxf(0.f, nx * lx + ny * ly + nz * lz);
    float ao = 1.f;
    for (int k = 1; k <= 5; k++) {
      float off = k * 0.04f;
      float d = de(px + nx * off, py + ny * off, pz + nz * off, tm);
      ao -= fmaxf(0.f, (off - d) * 1.6f / k);
    }
    ao = fminf(fmaxf(ao, 0.f), 1.f);
    float spec = powf(fmaxf(0.f, nx * (lx + fx) + ny * (ly + fy) + nz * (lz + fz)), 12.f);
    float fog = expf(-t * 0.12f);
    float sh = (0.08f + 0.7f * dif + 0.22f * spec) * ao * fog;
    int n = (int)(sh * 13.f);
    if (n < 0) n = 0;
    if (n > 12) n = 12;
    out[i] = " .,:;+=*#%@@WM"[n];
  }
  extern "C" void render(char* host, int w, int h, int tick) {
    static char* d; static int cap;
    int n = w * h;
    if (n > cap) { if (d) cudaFree(d); cudaMalloc(&d, n); cap = n; }
    trace<<<(n + 255) / 256, 256>>>(d, w, h, tick);
    cudaDeviceSynchronize();
    cudaMemcpy(host, d, n, cudaMemcpyDeviceToHost);
  }
CU

buf = Mvcc.buffer(w * h)
at_exit { print "\e[?25h" } if $stdout.tty?
print "\e[2J\e[?25l" if $stdout.tty?

($stdout.tty? ? 0.step : 1.times).each do |t|
  k.render(buf, w, h, t)
  print "\e[H" if $stdout.tty?
  puts buf.lines(w)
  sleep 0.04
end
