#!/usr/bin/env python3
"""Convert a Qwen3.5-MoE GPTQ-Int4 checkpoint into the flat .mvccq pack read by qwen.cu.

Only numpy is required (bf16 is handled as raw uint16). The output is a single file:

    "MVCCQ001" | u64 header_len | header JSON (padded to 64B) | 64B-aligned tensor blobs

Header JSON = {"config": {...}, "tensors": {name: {"off","nbytes","dtype","shape"}}}.

Tensor naming (per layer i):
    embed, lm_head, final_norm
    L{i}.rms1  L{i}.rms2                       bf16 [H]     (1 + w already folded in)
    L{i}.router                               bf16 [E, H]
    L{i}.shared.gate_up / .down / .gate       bf16 [2Is,H] / [H,Is] / [H]
    L{i}.exp.gate_up.q / .s                   u8 [E,2I,K/2] / f16 [E,2I,K/G]
    L{i}.exp.down.q / .s                      u8 [E,H,I/2]  / f16 [E,H,I/G]
    L{i}.gdn.qkv / .z / .ab / .out / .conv    bf16 [8192,H] / [4096,H] / [2*Hv,H] / [H,4096] / [8192,4]
    L{i}.gdn.A_log / .dt_bias / .norm         f32 [Hv] / f32 [Hv] / bf16 [Dv]
    L{i}.attn.qkv / .gate / .o                bf16 [(Hq+2Hkv)*D, H] (q rows de-interleaved) / [Hq*D,H] / [H,Hq*D]
    L{i}.attn.q_norm / .k_norm                bf16 [D] (1 + w folded in)

INT4 rows are packed along K: byte j of a row holds k=2j in the low nibble and k=2j+1 in the
high nibble, unsigned 0..15 with an implicit zero point of 8 (GPTQ sym): w = (q - 8) * scale.

Optional --dense int8 quantizes every bf16 projection matrix (GDN/attention/shared expert/lm_head)
to per-row symmetric int8 with a group size of 128 (name suffix .q8 / .s8).
"""
import argparse
import json
import mmap
import os
import struct
import sys
import time
import unicodedata

import numpy as np

# ------------------------------------------------------------------ safetensors (minimal, mmap)

DTYPE_ITEM = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "U8": 1, "I8": 1, "BOOL": 1}


class Shard:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        (n,) = struct.unpack("<Q", self.mm[:8])
        self.header = json.loads(self.mm[8 : 8 + n])
        self.base = 8 + n

    def tensor(self, name):
        meta = self.header[name]
        a, b = meta["data_offsets"]
        raw = np.frombuffer(self.mm, dtype=np.uint8, count=b - a, offset=self.base + a)
        dt = meta["dtype"]
        if dt == "BF16":
            arr = raw.view(np.uint16)
        elif dt == "F16":
            arr = raw.view(np.float16)
        elif dt == "F32":
            arr = raw.view(np.float32)
        elif dt == "I32":
            arr = raw.view(np.int32)
        elif dt == "I64":
            arr = raw.view(np.int64)
        else:
            raise ValueError(f"unsupported dtype {dt} for {name}")
        return arr.reshape(meta["shape"]), dt


class Checkpoint:
    def __init__(self, root):
        self.root = root
        idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
        self.weight_map = idx["weight_map"]
        self.shards = {}

    def has(self, name):
        return name in self.weight_map

    def get(self, name):
        shard = self.weight_map[name]
        if shard not in self.shards:
            self.shards[shard] = Shard(os.path.join(self.root, shard))
        return self.shards[shard].tensor(name)

    def close_all(self):
        for s in self.shards.values():
            try:
                s.mm.close()
            except BufferError:
                pass  # numpy views still alive; the mapping is released with the process
            s.f.close()
        self.shards = {}


# ------------------------------------------------------------------ bf16 helpers


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    # round to nearest even
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    return (u >> 16).astype(np.uint16)


def as_bf16(arr, dt):
    if dt == "BF16":
        return arr
    if dt == "F32":
        return f32_to_bf16(arr)
    if dt == "F16":
        return f32_to_bf16(arr.astype(np.float32))
    raise ValueError(dt)


def as_f32(arr, dt):
    if dt == "BF16":
        return bf16_to_f32(arr)
    return arr.astype(np.float32)


# ------------------------------------------------------------------ GPTQ int4 repack


def gptq_unpack(qweight, qzeros, scales, g_idx, group_size):
    """qweight int32 [K/8, N] -> q uint8 [N, K] (0..15), scales f16 [N, K/G]. Asserts sym zero=8."""
    Kp, N = qweight.shape
    K = Kp * 8
    if g_idx is not None:
        expect = np.arange(K, dtype=np.int32) // group_size
        if not np.array_equal(g_idx.astype(np.int32), expect):
            raise ValueError("desc_act / non-identity g_idx not supported")
    shifts = (np.arange(8, dtype=np.uint32) * 4)[None, :, None]
    q = ((qweight.view(np.uint32)[:, None, :] >> shifts) & 0xF).astype(np.uint8)  # [K/8, 8, N]
    q = q.reshape(K, N).T  # [N, K]
    # zeros: GPTQ v1 stores (zero - 1) -> 7; v2 (gptqmodel) stores the zero itself -> 8. Both mean sym.
    zshift = (np.arange(8, dtype=np.uint32) * 4)[None, None, :]
    z = ((qzeros.view(np.uint32)[:, :, None] >> zshift) & 0xF).astype(np.uint8)  # [K/G, N/8, 8]
    z = z.reshape(qzeros.shape[0], -1)  # [K/G, N]
    if not (np.all(z == 7) or np.all(z == 8)):
        raise ValueError("only symmetric GPTQ (zero point 8) is supported")
    s = scales.T.astype(np.float16)  # [N, K/G]
    return np.ascontiguousarray(q), np.ascontiguousarray(s)


def pack_nibbles(q):
    """q uint8 [N, K] -> [N, K/2] with even k in low nibble."""
    return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)


# ------------------------------------------------------------------ dense int8 (optional)


def quant_int8_rows(w_bf16, group=128):
    """bf16 uint16 [N, K] -> int8 [N, K], f16 scales [N, K/G] (symmetric per (row, group))."""
    N, K = w_bf16.shape
    f = bf16_to_f32(w_bf16).reshape(N, K // group, group)
    amax = np.abs(f).max(axis=2, keepdims=True)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(f / scale), -127, 127).astype(np.int8).reshape(N, K)
    return q, scale.reshape(N, K // group).astype(np.float16)


# ------------------------------------------------------------------ writer


class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.tensors = {}
        self.off = 0
        self.header_reserve = 1 << 20  # 1 MiB header slot; rewritten at close
        self.f.write(b"\0" * self.header_reserve)
        self.off = self.header_reserve

    def add(self, name, arr, dtype, shape=None):
        arr = np.ascontiguousarray(arr)
        pad = (-self.off) % 64
        if pad:
            self.f.write(b"\0" * pad)
            self.off += pad
        nbytes = arr.nbytes
        self.tensors[name] = {
            "off": self.off,
            "nbytes": nbytes,
            "dtype": dtype,
            "shape": list(shape if shape is not None else arr.shape),
        }
        self.f.write(arr.tobytes())
        self.off += nbytes

    def close(self, config):
        hdr = json.dumps({"config": config, "tensors": self.tensors}, separators=(",", ":")).encode()
        if 16 + len(hdr) > self.header_reserve:
            raise RuntimeError("header too large")
        self.f.seek(0)
        self.f.write(b"MVCCQ001")
        self.f.write(struct.pack("<Q", len(hdr)))
        self.f.write(hdr)
        self.f.close()


# ------------------------------------------------------------------ tokenizer export


def byte_decoder():
    """GPT-2 byte-level BPE unicode->byte map."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def unicode_ranges(pred):
    """Compress code points satisfying pred into sorted [lo, hi] inclusive ranges."""
    ranges = []
    start = None
    for cp in range(0x110000):
        ok = pred(cp)
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            ranges.append((start, cp - 1))
            start = None
    if start is not None:
        ranges.append((start, 0x10FFFF))
    return ranges


def export_tokenizer(model_dir, out_path):
    tok = json.load(open(os.path.join(model_dir, "tokenizer.json")))
    assert tok["model"]["type"] == "BPE"
    dec = byte_decoder()
    vocab = tok["model"]["vocab"]
    merges = tok["model"]["merges"]
    added = tok["added_tokens"]
    size = max(max(vocab.values()), max(a["id"] for a in added)) + 1
    pieces = [b""] * size
    for tokstr, tid in vocab.items():
        pieces[tid] = bytes(dec[ch] for ch in tokstr)
    special = []
    for a in added:
        pieces[a["id"]] = a["content"].encode()
        special.append((a["id"], a["content"]))

    def cat(cp):
        try:
            return unicodedata.category(chr(cp))
        except ValueError:
            return "Cn"

    letters = unicode_ranges(lambda cp: cat(cp)[0] == "L")
    marks = unicode_ranges(lambda cp: cat(cp)[0] == "M")
    numbers = unicode_ranges(lambda cp: cat(cp)[0] == "N")
    # Unicode White_Space (what \s means in the tokenizers regex engine)
    white = set(range(9, 14)) | {0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000} | set(range(0x2000, 0x200B))
    spaces = unicode_ranges(lambda cp: cp in white)

    with open(out_path, "wb") as f:
        f.write(b"MVCCT001")
        f.write(struct.pack("<IIII", size, len(merges), len(special), 0))
        for p in pieces:
            f.write(struct.pack("<H", len(p)))
            f.write(p)
        for m in merges:
            if isinstance(m, str):
                a, b = m.split(" ")
            else:
                a, b = m
            f.write(struct.pack("<III", vocab[a], vocab[b], vocab[a + b]))
        for tid, s in special:
            e = s.encode()
            f.write(struct.pack("<IH", tid, len(e)))
            f.write(e)
        for table in (letters, marks, numbers, spaces):
            f.write(struct.pack("<I", len(table)))
            f.write(np.array(table, dtype=np.uint32).tobytes())
    print(f"tokenizer: {size} tokens, {len(merges)} merges, {len(special)} special -> {out_path}")


# ------------------------------------------------------------------ main conversion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--dense", choices=["bf16", "int8"], default="bf16", help="dense projection storage")
    ap.add_argument("--lm-head", choices=["bf16", "int8"], default=None, help="defaults to --dense")
    ap.add_argument("--layers", type=int, default=0, help="limit layers (debug)")
    ap.add_argument("--tokenizer-out", default=None)
    args = ap.parse_args()
    lm_mode = args.lm_head or args.dense

    cfg_all = json.load(open(os.path.join(args.model_dir, "config.json")))
    tc = cfg_all["text_config"]
    qc = cfg_all.get("quantization_config", {})
    G = int(qc.get("group_size", 128))
    H = tc["hidden_size"]
    L = args.layers or tc["num_hidden_layers"]
    E = tc["num_experts"]
    I = tc["moe_intermediate_size"]
    Is = tc["shared_expert_intermediate_size"]
    Hq = tc["num_attention_heads"]
    Hkv = tc["num_key_value_heads"]
    D = tc["head_dim"]
    Hk = tc["linear_num_key_heads"]
    Hv = tc["linear_num_value_heads"]
    Dk = tc["linear_key_head_dim"]
    Dv = tc["linear_value_head_dim"]
    V = tc["vocab_size"]
    kinds = [1 if t == "linear_attention" else 0 for t in tc["layer_types"][:L]]
    rope = tc["rope_parameters"]
    config = {
        "hidden": H, "layers": L, "experts": E, "topk": tc["num_experts_per_tok"], "inter": I, "shared_inter": Is,
        "q_heads": Hq, "kv_heads": Hkv, "head_dim": D, "gdn_k_heads": Hk, "gdn_v_heads": Hv, "gdn_k_dim": Dk,
        "gdn_v_dim": Dv, "conv_k": tc["linear_conv_kernel_dim"], "vocab": V, "eps": tc["rms_norm_eps"],
        "rope_theta": rope["rope_theta"], "rope_dim": int(D * rope.get("partial_rotary_factor", 1.0)),
        "layer_kinds": kinds, "group": G, "eos": [tc["eos_token_id"], 248044], "dense": args.dense, "lm_head": lm_mode,
        "attn_gate": bool(tc.get("attn_output_gate", False)),
    }
    print("config:", json.dumps(config))

    ck = Checkpoint(args.model_dir)
    w = Writer(args.out)
    P = "model.language_model."

    def dense(name, key, rows_perm=None, mode=None):
        arr, dt = ck.get(key)
        b = as_bf16(arr, dt)
        if rows_perm is not None:
            b = b[rows_perm]
        mode = mode or args.dense
        if mode == "int8":
            q, s = quant_int8_rows(b)
            w.add(name + ".q8", q, "i8")
            w.add(name + ".s8", s, "f16")
        else:
            w.add(name, b, "bf16")

    def norm_plus_one(name, key, dim):
        arr, dt = ck.get(key)
        f = as_f32(arr, dt).reshape(-1)
        assert f.shape[0] == dim, (key, f.shape)
        w.add(name, f32_to_bf16(f + 1.0), "bf16")

    t0 = time.time()
    embed, dt = ck.get(P + "embed_tokens.weight")
    w.add("embed", as_bf16(embed, dt), "bf16")
    dense("lm_head", "lm_head.weight", mode=lm_mode)
    norm_plus_one("final_norm", P + "norm.weight", H)
    print(f"embed/lm_head/norm done ({time.time()-t0:.1f}s)")

    for li in range(L):
        tl = time.time()
        pre = f"{P}layers.{li}."
        nm = f"L{li}."
        norm_plus_one(nm + "rms1", pre + "input_layernorm.weight", H)
        norm_plus_one(nm + "rms2", pre + "post_attention_layernorm.weight", H)
        arr, dt = ck.get(pre + "mlp.gate.weight")
        w.add(nm + "router", as_bf16(arr, dt), "bf16")
        # shared expert (bf16 in the GPTQ checkpoint)
        g, dtg = ck.get(pre + "mlp.shared_expert.gate_proj.weight")
        u, dtu = ck.get(pre + "mlp.shared_expert.up_proj.weight")
        gu = np.concatenate([as_bf16(g, dtg), as_bf16(u, dtu)], axis=0)
        if args.dense == "int8":
            q, s = quant_int8_rows(gu)
            w.add(nm + "shared.gate_up.q8", q, "i8")
            w.add(nm + "shared.gate_up.s8", s, "f16")
        else:
            w.add(nm + "shared.gate_up", gu, "bf16")
        dense(nm + "shared.down", pre + "mlp.shared_expert.down_proj.weight")
        arr, dt = ck.get(pre + "mlp.shared_expert_gate.weight")
        w.add(nm + "shared.gate", as_bf16(arr, dt).reshape(-1), "bf16")
        # routed experts: int4
        gu_q = np.empty((E, 2 * I, H // 2), dtype=np.uint8)
        gu_s = np.empty((E, 2 * I, H // G), dtype=np.float16)
        dn_q = np.empty((E, H, I // 2), dtype=np.uint8)
        dn_s = np.empty((E, H, I // G), dtype=np.float16)
        for e in range(E):
            ep = f"{pre}mlp.experts.{e}."
            for j, proj in enumerate(("gate_proj", "up_proj")):
                qw, _ = ck.get(ep + proj + ".qweight")
                qz, _ = ck.get(ep + proj + ".qzeros")
                sc, _ = ck.get(ep + proj + ".scales")
                gi = ck.get(ep + proj + ".g_idx")[0] if ck.has(ep + proj + ".g_idx") else None
                q, s = gptq_unpack(qw, qz, sc, gi, G)
                assert q.shape == (I, H), q.shape
                gu_q[e, j * I : (j + 1) * I] = pack_nibbles(q)
                gu_s[e, j * I : (j + 1) * I] = s
            qw, _ = ck.get(ep + "down_proj.qweight")
            qz, _ = ck.get(ep + "down_proj.qzeros")
            sc, _ = ck.get(ep + "down_proj.scales")
            gi = ck.get(ep + "down_proj.g_idx")[0] if ck.has(ep + "down_proj.g_idx") else None
            q, s = gptq_unpack(qw, qz, sc, gi, G)
            assert q.shape == (H, I), q.shape
            dn_q[e] = pack_nibbles(q)
            dn_s[e] = s
        w.add(nm + "exp.gate_up.q", gu_q, "u8")
        w.add(nm + "exp.gate_up.s", gu_s, "f16")
        w.add(nm + "exp.down.q", dn_q, "u8")
        w.add(nm + "exp.down.s", dn_s, "f16")
        if kinds[li] == 1:
            gp = pre + "linear_attn."
            dense(nm + "gdn.qkv", gp + "in_proj_qkv.weight")
            dense(nm + "gdn.z", gp + "in_proj_z.weight")
            a, dta = ck.get(gp + "in_proj_a.weight")
            b, dtb = ck.get(gp + "in_proj_b.weight")
            w.add(nm + "gdn.ab", np.concatenate([as_bf16(a, dta), as_bf16(b, dtb)], axis=0), "bf16")
            dense(nm + "gdn.out", gp + "out_proj.weight")
            conv, dtc = ck.get(gp + "conv1d.weight")
            conv = as_bf16(conv, dtc).reshape(-1, tc["linear_conv_kernel_dim"])
            assert conv.shape[0] == 2 * Hk * Dk + Hv * Dv
            w.add(nm + "gdn.conv", conv, "bf16")
            al, dt = ck.get(gp + "A_log")
            w.add(nm + "gdn.A_log", as_f32(al, dt).reshape(-1), "f32")
            db, dt = ck.get(gp + "dt_bias")
            w.add(nm + "gdn.dt_bias", as_f32(db, dt).reshape(-1), "f32")
            nw, dt = ck.get(gp + "norm.weight")
            w.add(nm + "gdn.norm", as_bf16(nw, dt).reshape(-1), "bf16")
        else:
            ap_ = pre + "self_attn."
            qw, dtq = ck.get(ap_ + "q_proj.weight")
            qw = as_bf16(qw, dtq)
            if config["attn_gate"]:
                assert qw.shape[0] == Hq * 2 * D, qw.shape
                qw3 = qw.reshape(Hq, 2, D, H)
                qrows = qw3[:, 0].reshape(Hq * D, H)
                grows = qw3[:, 1].reshape(Hq * D, H)
            else:
                qrows, grows = qw, None
            kw, dtk = ck.get(ap_ + "k_proj.weight")
            vw, dtv = ck.get(ap_ + "v_proj.weight")
            qkv = np.concatenate([qrows, as_bf16(kw, dtk), as_bf16(vw, dtv)], axis=0)
            if args.dense == "int8":
                q, s = quant_int8_rows(qkv)
                w.add(nm + "attn.qkv.q8", q, "i8")
                w.add(nm + "attn.qkv.s8", s, "f16")
                if grows is not None:
                    q, s = quant_int8_rows(np.ascontiguousarray(grows))
                    w.add(nm + "attn.gate.q8", q, "i8")
                    w.add(nm + "attn.gate.s8", s, "f16")
            else:
                w.add(nm + "attn.qkv", qkv, "bf16")
                if grows is not None:
                    w.add(nm + "attn.gate", grows, "bf16")
            dense(nm + "attn.o", ap_ + "o_proj.weight")
            norm_plus_one(nm + "attn.q_norm", ap_ + "q_norm.weight", D)
            norm_plus_one(nm + "attn.k_norm", ap_ + "k_norm.weight", D)
        print(f"layer {li} ({'gdn' if kinds[li] else 'attn'}) done in {time.time()-tl:.1f}s, total {w.off/1e9:.2f} GB")
        sys.stdout.flush()
    w.close(config)
    ck.close_all()
    print(f"wrote {args.out}: {w.off/1e9:.2f} GB in {time.time()-t0:.0f}s")
    if args.tokenizer_out:
        export_tokenizer(args.model_dir, args.tokenizer_out)


if __name__ == "__main__":
    main()
