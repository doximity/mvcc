//! Launch assumptions of recovered kernels.
//!
//! The compiler proves a retiled body correct under conditions on the kernel's arguments alone (which ring slot a
//! K iteration reads was refilled before it, addresses derived from an argument are non-null, ...). It serializes
//! them as s-expressions over the parameter block; the runtime evaluates them at every launch and runs the exact
//! twin when one fails or cannot be evaluated.
//!
//! Grammar (integers are i64; `(wrap W e)` reduces e to W bits, sign-extended):
//!   e := <int> | (arg I) | (+ (* c a..)..) | (wrap W e) | (CMP W a b) | (ite c a b) | (and a..) | (or a..) | (xor a b)
//!      | (shl a b) | (lshr a b) | (ashr a b) | (sdiv a b) | (udiv a b) | (srem a b) | (urem a b) | (mul a b) | (add a b)
//!      | (sub a b) | (sext W a) | (zext W a) | (trunc a) | (min a b) | (max a b) | (=> a b) | (false)
//!   CMP := eq | ne | slt | sle | sgt | sge | ult | ule | ugt | uge   (W: the operands' width)

use crate::module::ParamAbi;

enum Sx { Int(i64), List(Vec<Sx>), Atom(String) }

fn parse(s: &str) -> Option<Sx> {
    let mut toks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '(' | ')' => { if !cur.is_empty() { toks.push(std::mem::take(&mut cur)); } toks.push(ch.to_string()); }
            c if c.is_whitespace() => { if !cur.is_empty() { toks.push(std::mem::take(&mut cur)); } }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() { toks.push(cur); }
    fn node(toks: &[String], i: &mut usize) -> Option<Sx> {
        let t = toks.get(*i)?; *i += 1;
        if t == "(" {
            let mut v = Vec::new();
            loop {
                if toks.get(*i)? == ")" { *i += 1; return Some(Sx::List(v)); }
                v.push(node(toks, i)?);
            }
        }
        if t == ")" { return None; }
        if let Ok(n) = t.parse::<i64>() { return Some(Sx::Int(n)); }
        Some(Sx::Atom(t.clone()))
    }
    let mut i = 0;
    let r = node(&toks, &mut i)?;
    if i != toks.len() { return None; }
    Some(r)
}

fn wrap(v: i64, w: i64) -> Option<i64> {
    match w { 1 => Some(v & 1), 8 => Some(v as i8 as i64), 16 => Some(v as i16 as i64), 32 => Some(v as i32 as i64), 64 => Some(v), _ => None }
}
fn unsigned(v: i64, w: i64) -> Option<u64> {
    match w { 1 => Some((v & 1) as u64), 8 => Some(v as u8 as u64), 16 => Some(v as u16 as u64), 32 => Some(v as u32 as u64), 64 => Some(v as u64), _ => None }
}

/// The argument at byte offset `p.offset` of the parameter block as an integer (pointers as their address).
fn arg(params: &[u8], abi: &[ParamAbi], idx: usize) -> Option<i64> {
    let p = abi.get(idx)?;
    let b = params.get(p.offset as usize..(p.offset + p.size) as usize)?;
    Some(match p.size {
        1 => b[0] as i8 as i64,
        2 => i16::from_le_bytes([b[0], b[1]]) as i64,
        4 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64,
        8 => i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        _ => return None,
    })
}

fn eval(e: &Sx, params: &[u8], abi: &[ParamAbi]) -> Option<i64> {
    let ev = |x: &Sx| eval(x, params, abi);
    match e {
        Sx::Int(n) => Some(*n),
        Sx::Atom(_) => None,
        Sx::List(v) => {
            let Some(Sx::Atom(op)) = v.first() else { return None };
            let a = &v[1..];
            match op.as_str() {
                "false" => Some(0),
                "arg" => { let Sx::Int(i) = a.first()? else { return None }; arg(params, abi, *i as usize) }
                "+" => { let mut s: i64 = 0; for t in a { s = s.wrapping_add(ev(t)?); } Some(s) }
                "*" => { let mut p: i64 = 1; for t in a { p = p.wrapping_mul(ev(t)?); } Some(p) }
                "wrap" => wrap(ev(a.get(1)?)?, ev(a.first()?)?),
                "ite" => Some(if ev(a.first()?)? != 0 { ev(a.get(1)?)? } else { ev(a.get(2)?)? }),
                "and" => { for t in a { if ev(t)? == 0 { return Some(0); } } Some(1) }
                "or" => { for t in a { if ev(t)? != 0 { return Some(1); } } Some(0) }
                "=>" => Some(if ev(a.first()?)? == 0 || ev(a.get(1)?)? != 0 { 1 } else { 0 }),
                "xor" => Some(ev(a.first()?)? ^ ev(a.get(1)?)?),
                "add" => Some(ev(a.first()?)?.wrapping_add(ev(a.get(1)?)?)),
                "sub" => Some(ev(a.first()?)?.wrapping_sub(ev(a.get(1)?)?)),
                "mul" => Some(ev(a.first()?)?.wrapping_mul(ev(a.get(1)?)?)),
                "shl" => { let s = ev(a.get(1)?)?; if !(0..64).contains(&s) { return None; } Some(ev(a.first()?)? << s) }
                "lshr" => { let s = ev(a.get(1)?)?; if !(0..64).contains(&s) { return None; } Some(((ev(a.first()?)? as u64) >> s) as i64) }
                "ashr" => { let s = ev(a.get(1)?)?; if !(0..64).contains(&s) { return None; } Some(ev(a.first()?)? >> s) }
                "sdiv" => { let d = ev(a.get(1)?)?; if d == 0 { return None; } Some(ev(a.first()?)?.wrapping_div(d)) }
                "srem" => { let d = ev(a.get(1)?)?; if d == 0 { return None; } Some(ev(a.first()?)?.wrapping_rem(d)) }
                "udiv" => { let d = ev(a.get(1)?)? as u64; if d == 0 { return None; } Some(((ev(a.first()?)? as u64) / d) as i64) }
                "urem" => { let d = ev(a.get(1)?)? as u64; if d == 0 { return None; } Some(((ev(a.first()?)? as u64) % d) as i64) }
                "min" => Some(ev(a.first()?)?.min(ev(a.get(1)?)?)),
                "max" => Some(ev(a.first()?)?.max(ev(a.get(1)?)?)),
                "sext" => wrap(ev(a.get(1)?)?, ev(a.first()?)?),
                "zext" => Some(unsigned(ev(a.get(1)?)?, ev(a.first()?)?)? as i64),
                "trunc" => ev(a.first()?),
                "eq" | "ne" | "slt" | "sle" | "sgt" | "sge" | "ult" | "ule" | "ugt" | "uge" => {
                    let w = ev(a.first()?)?;
                    let (x, y) = (wrap(ev(a.get(1)?)?, w)?, wrap(ev(a.get(2)?)?, w)?);
                    let (ux, uy) = (unsigned(x, w)?, unsigned(y, w)?);
                    let r = match op.as_str() {
                        "eq" => x == y, "ne" => x != y, "slt" => x < y, "sle" => x <= y, "sgt" => x > y, "sge" => x >= y,
                        "ult" => ux < uy, "ule" => ux <= uy, "ugt" => ux > uy, _ => ux >= uy,
                    };
                    Some(r as i64)
                }
                _ => None,
            }
        }
    }
}

/// Whether every assumption holds for this parameter block. `Err(which)` names the first that fails or cannot be
/// evaluated (a malformed expression is a failed assumption: the twin runs).
pub fn check(assume: &[String], params: &[u8], abi: &[ParamAbi]) -> Result<(), String> {
    // parsed once per distinct expression: a decode kernel's assumptions run to a few KB of text, and a launch
    // is a few microseconds of host time (README.md)
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<Option<Sx>>>>> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    for s in assume {
        let e = {
            let mut c = cache.lock().unwrap_or_else(|p| p.into_inner());
            match c.get(s) { Some(x) => x.clone(), None => { let x = std::sync::Arc::new(parse(s)); c.insert(s.clone(), x.clone()); x } }
        };
        match e.as_ref().as_ref().and_then(|e| eval(e, params, abi)) {
            Some(v) if v != 0 => {}
            Some(_) => return Err(s.clone()),
            None => return Err(format!("{} (not evaluable)", s)),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn abi() -> Vec<ParamAbi> { vec![ParamAbi { kind: "ptr".into(), size: 8, align: 8, offset: 0 }, ParamAbi { kind: "i32".into(), size: 4, align: 4, offset: 8 }] }
    fn params(p: u64, k: i32) -> Vec<u8> { let mut v = p.to_le_bytes().to_vec(); v.extend_from_slice(&k.to_le_bytes()); v }
    #[test]
    fn evaluates_ring_refill_implication() {
        // K > 255 and K/256 != 1  =>  K > 511 (a tautology over the integers: the twin never runs for it)
        let e = "(=> (and (sgt 32 (wrap 32 (+ (* -255) (* 1 (arg 1)))) 0) (ne 32 1 (wrap 32 (+ (* 1 (wrap 32 (sdiv (wrap 32 (+ (* 1 (arg 1)))) 256))))))) (sgt 32 (wrap 32 (+ (* -511) (* 1 (arg 1)))) 0))".to_string();
        for k in [0, 128, 255, 256, 511, 512, 4096, 7168] { assert!(check(&[e.clone()], &params(1, k), &abi()).is_ok(), "K={}", k); }
    }
    #[test]
    fn non_null_and_sign() {
        let nn = "(ne 64 (+ (* 1 (arg 0))) 0)".to_string();
        assert!(check(&[nn.clone()], &params(0x1000, 1), &abi()).is_ok());
        assert!(check(&[nn], &params(0, 1), &abi()).is_err());
        let pos = "(sgt 32 (wrap 32 (+ (* 1) (* 1 (arg 1)))) 0)".to_string();
        assert!(check(&[pos.clone()], &params(1, 0), &abi()).is_ok());
        assert!(check(&[pos], &params(1, -1), &abi()).is_err());
    }
    #[test]
    fn malformed_fails_closed() {
        assert!(check(&["(false)".to_string()], &params(1, 1), &abi()).is_err());
        assert!(check(&["(frob 1 2".to_string()], &params(1, 1), &abi()).is_err());
        assert!(check(&["(arg 7)".to_string()], &params(1, 1), &abi()).is_err());
    }
}
