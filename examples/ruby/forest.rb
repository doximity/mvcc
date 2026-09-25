# Requires the mvcc gem (README.md, "Install").
require "mvcc"

w, h = 64, 40
k = Mvcc.compile(<<~'CU')
  #include <cuda_runtime.h>
  #include <math.h>
  __device__ float fractf(float x) { return x - floorf(x); }
  __device__ float hash(float n) { return fractf(sinf(n) * 43758.5453123f); }
  __device__ float n2(float x, float z) {
    float ix = floorf(x), iz = floorf(z), fx = x - ix, fz = z - iz;
    fx = fx * fx * (3.f - 2.f * fx);
    fz = fz * fz * (3.f - 2.f * fz);
    float a = hash(ix + iz * 57.f), b = hash(ix + 1.f + iz * 57.f);
    float c = hash(ix + (iz + 1.f) * 57.f), d = hash(ix + 1.f + (iz + 1.f) * 57.f);
    return a + (b - a) * fx + (c - a) * fz + (a - b - c + d) * fx * fz;
  }
  __device__ float n3(float x, float y, float z) {
    float a = n2(x, z), b = n2(x + 19.1f, y + 7.3f);
    return a + (b - a) * fractf(y * 0.5f + a);
  }
  __device__ unsigned char sat(float x) {
    return (unsigned char)(int)(fminf(fmaxf(x, 0.f), 1.f) * 255.f);
  }
  __device__ float cap(float x, float y, float z,
                      float ax, float ay, float az, float bx, float by, float bz, float r) {
    float pax = x - ax, pay = y - ay, paz = z - az;
    float bax = bx - ax, bay = by - ay, baz = bz - az;
    float den = bax * bax + bay * bay + baz * baz + 1e-8f;
    float t = fminf(fmaxf((pax * bax + pay * bay + paz * baz) / den, 0.f), 1.f);
    float dx = pax - bax * t, dy = pay - bay * t, dz = paz - baz * t;
    return sqrtf(dx * dx + dy * dy + dz * dz) - r;
  }
  __device__ float tree(float x, float y, float z, float seed, int* part) {
    float hgt = 1.45f + 0.7f * hash(seed);
    x -= (hash(seed + 1.f) - 0.5f) * 0.08f * y;
    z -= (hash(seed + 1.4f) - 0.5f) * 0.06f * y;
    float flare = 0.085f + 0.05f * expf(-y * 8.f);
    float ty = fminf(fmaxf(y / (hgt * 0.7f), 0.f), 1.f);
    float tr = flare * (1.f - 0.62f * ty);
    float wood = sqrtf(x * x + z * z) - tr + 0.01f * n3(x * 22.f, y * 14.f, z * 22.f);
    wood = fmaxf(wood, -y - 0.03f);
    wood = fmaxf(wood, y - hgt * 0.72f);
    for (int k = 0; k < 3; k++) {
      float ra = hash(seed + 30.f + k) * 6.2832f;
      float rl = 0.16f + 0.08f * hash(seed + 40.f + k);
      wood = fminf(wood, cap(x, y, z, 0.f, 0.02f, 0.f, cosf(ra) * rl, 0.01f, sinf(ra) * rl, 0.03f));
    }
    int pine = hash(seed + 2.f) > 0.38f;
    float can = 1e5f;
    if (pine) {
      for (int i = 0; i < 7; i++) {
        float q = i / 6.f;
        float oy = hgt * (0.28f + 0.1f * i);
        float rad = 0.58f * (1.f - 0.78f * q);
        float sx = x, sy = y - oy, sz = z;
        float cone = sqrtf(sx * sx + sz * sz) - rad * fmaxf(0.f, 1.f - sy / (0.22f + 0.06f * i));
        cone = fmaxf(cone, -0.12f - sy);
        cone = fmaxf(cone, sy - 0.1f);
        can = fminf(can, cone);
      }
    } else {
      for (int i = 0; i < 9; i++) {
        float ox = (hash(seed + i * 3.f) - 0.5f) * 0.55f;
        float oy = hgt * (0.42f + 0.08f * i) + hash(seed + i * 3.f + 1.f) * 0.18f;
        float oz = (hash(seed + i * 3.f + 2.f) - 0.5f) * 0.55f;
        float rad = 0.2f + 0.16f * hash(seed + 18.f + i) * (1.f - i * 0.04f);
        float sx = x - ox, sy = y - oy, sz = z - oz;
        can = fminf(can, sqrtf(sx * sx + sy * sy * 1.15f + sz * sz) - rad);
      }
    }
    for (int i = 0; i < 6; i++) {
      float a = hash(seed + 70.f + i) * 6.2832f;
      float yh = hgt * (0.28f + 0.08f * i);
      float len = 0.22f + 0.18f * hash(seed + 80.f + i);
      float up = yh + 0.12f + 0.1f * hash(seed + 90.f + i);
      wood = fminf(wood, cap(x, y, z, 0.f, yh, 0.f, cosf(a) * len, up, sinf(a) * len, 0.018f));
    }
    can += 0.012f * n3(x * 7.f, y * 7.f, z * 7.f);
    if (wood < can) { *part = 1; return wood; }
    *part = 2;
    return can;
  }
  __device__ float map(float x, float y, float z, int* mat) {
    float g = y - 0.035f * n2(x * 2.2f, z * 2.2f);
    *mat = 0;
    float d = g;
    const float CS = 2.8f;
    float ix0 = floorf(x / CS), iz0 = floorf(z / CS);
    for (int dx = -1; dx <= 1; dx++) {
      for (int dz = -1; dz <= 1; dz++) {
        float cx = (ix0 + dx) * CS, cz = (iz0 + dz) * CS;
        float seed = cx * 3.17f + cz * 19.73f;
        float tx = cx + (hash(seed) - 0.5f) * 1.15f;
        float tz = cz + (hash(seed + 1.f) - 0.5f) * 1.15f;
        float s = 0.8f + 0.32f * hash(seed + 5.f);
        int part = 0;
        float td = tree((x - tx) / s, y / s, (z - tz) / s, seed, &part) * s;
        if (td < d) { d = td; *mat = part; }
      }
    }
    return d;
  }
  __global__ void trace(unsigned char* rgb, int w, int h,
                       float ox, float oy, float oz, float yaw, float pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h) return;
    float u = ((i % w) + 0.5f) / w * 2.f - 1.f;
    float v = 1.f - ((i / w) + 0.5f) / h * 2.f;
    u *= (float)w / (float)h * 0.52f;
    float cp = cosf(pitch), sp = sinf(pitch), cy = cosf(yaw), sy = sinf(yaw);
    float fx = sy * cp, fy = sp, fz = cy * cp;
    float rx = -fz, rz = fx, rl = sqrtf(rx * rx + rz * rz) + 1e-8f;
    rx /= rl; rz /= rl;
    float ux = -rz * fy, uy = rz * fx - rx * fz, uz = rx * fy;
    float dx = fx + u * rx + v * ux, dy = fy + v * uy, dz = fz + u * rz + v * uz;
    float dl = sqrtf(dx * dx + dy * dy + dz * dz);
    dx /= dl; dy /= dl; dz /= dl;
    float lx = -0.35f, ly = 0.78f, lz = 0.52f;
    float ll = sqrtf(lx * lx + ly * ly + lz * lz);
    lx /= ll; ly /= ll; lz /= ll;
    float t = 0.f;
    int hit = 0, mat = 0;
    for (int s = 0; s < 40; s++) {
      float d = map(ox + t * dx, oy + t * dy, oz + t * dz, &mat);
      if (d < 0.0015f) { hit = 1; break; }
      t += d * 0.85f;
      if (t > 18.f) break;
    }
    float cr = 0.45f + 0.35f * v, cg = 0.62f + 0.2f * v, cb = 0.85f;
    float sun = powf(fmaxf(dx * lx + dy * ly + dz * lz, 0.f), 64.f);
    cr += sun * 1.1f; cg += sun * 0.85f; cb += sun * 0.45f;
    if (hit) {
      float px = ox + t * dx, py = oy + t * dy, pz = oz + t * dz, e = 0.012f;
      int m;
      float nx = map(px + e, py, pz, &m) - map(px - e, py, pz, &m);
      float ny = map(px, py + e, pz, &m) - map(px, py - e, pz, &m);
      float nz = map(px, py, pz + e, &m) - map(px, py, pz - e, &m);
      float nl = sqrtf(nx * nx + ny * ny + nz * nz) + 1e-8f;
      nx /= nl; ny /= nl; nz /= nl;
      map(px, py, pz, &mat);
      float dif = fmaxf(0.f, nx * lx + ny * ly + nz * lz);
      float sh = 1.f, st = 0.05f;
      for (int k = 0; k < 2; k++) {
        float sd = map(px + lx * st, py + ly * st, pz + lz * st, &m);
        sh = fminf(sh, 7.f * sd / st);
        st += fmaxf(sd, 0.07f);
      }
      sh = fmaxf(sh, 0.f);
      float back = fmaxf(0.f, -(nx * lx + ny * ly + nz * lz));
      float gvar = n2(px * 6.f, pz * 6.f);
      if (mat == 0) {
        cr = 0.22f + 0.18f * gvar; cg = 0.32f + 0.14f * gvar; cb = 0.10f;
      } else if (mat == 1) {
        cr = 0.28f; cg = 0.16f; cb = 0.08f;
        cr *= 0.55f + 0.2f * n2(px * 14.f, py * 18.f);
      } else {
        float leaf = 0.25f + 0.55f * n3(px * 18.f, py * 16.f, pz * 18.f);
        cr = 0.06f + 0.16f * leaf; cg = 0.14f + 0.42f * leaf; cb = 0.05f + 0.08f * leaf;
        cr += back * 0.35f; cg += back * 0.55f; cb += back * 0.12f;
      }
      float amb = 0.18f + 0.12f * ny;
      float shade = amb + 0.82f * dif * sh;
      cr *= shade; cg *= shade; cb *= shade;
      float fog = expf(-t * 0.09f);
      cr = cr * fog + 0.55f * (1.f - fog);
      cg = cg * fog + 0.68f * (1.f - fog);
      cb = cb * fog + 0.78f * (1.f - fog);
    }
    rgb[i * 3] = sat(cr);
    rgb[i * 3 + 1] = sat(cg);
    rgb[i * 3 + 2] = sat(cb);
  }
  extern "C" void render(unsigned char* host, int w, int h,
                        float ox, float oy, float oz, float yaw, float pitch) {
    static unsigned char* d; static int cap;
    int n = w * h * 3;
    if (n > cap) { if (d) cudaFree(d); cudaMalloc(&d, n); cap = n; }
    trace<<<(w * h + 255) / 256, 256>>>(d, w, h, ox, oy, oz, yaw, pitch);
    cudaDeviceSynchronize();
    cudaMemcpy(host, d, n, cudaMemcpyDeviceToHost);
  }
CU

buf = Mvcc.buffer(w * h * 3)
x = 1.3
y = 0.95
z = 0.8
yaw = 0.0
pitch = -0.06
speed = 10
steer = 1.55
hold = 0.22
live = $stdout.tty?
quit = false
held = { r: 0.0, l: 0.0, u: 0.0, d: 0.0 }
lock = Mutex.new
now = -> { Process.clock_gettime(Process::CLOCK_MONOTONIC) }

if live
  require "io/console"
  STDIN.raw!
  at_exit do
    STDIN.cooked! rescue nil
    print "\e[?25h"
  end
  print "\e[2J\e[?25l"
  Thread.new do
    loop do
      chunk = STDIN.readpartial(64)
      t = now.call
      lock.synchronize do
        quit = true if chunk.include?("q") || chunk.include?("\u0003")
        held[:r] = t if chunk.match?(/\e(?:\[|O)C/)
        held[:l] = t if chunk.match?(/\e(?:\[|O)D/)
        held[:u] = t if chunk.match?(/\e(?:\[|O)A/)
        held[:d] = t if chunk.match?(/\e(?:\[|O)B/)
      end
    end
  rescue EOFError, IOError
  end
end

prev = now.call
loop do
  t = now.call
  dt = (t - prev).clamp(1.0 / 120.0, 0.12)
  prev = t
  if live
    lock.synchronize do
      yaw += steer * dt if t - held[:r] < hold
      yaw -= steer * dt if t - held[:l] < hold
      pitch += steer * dt if t - held[:u] < hold
      pitch -= steer * dt if t - held[:d] < hold
    end
    break if lock.synchronize { quit }
    pitch = pitch.clamp(-1.15, 1.15)
  end
  cp = Math.cos(pitch)
  x += Math.sin(yaw) * cp * speed * dt
  y += Math.sin(pitch) * speed * dt
  z += Math.cos(yaw) * cp * speed * dt
  y = y.clamp(0.42, 7.0)
  k.render(buf, w, h, x, y, z, yaw, pitch)
  frame = buf.halfblocks(w)
  if live
    print "\e[H#{frame.gsub("\n", "\r\n")}\r\n"
  else
    puts frame
  end
  break unless live
end

