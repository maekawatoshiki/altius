use std::{
    mem::transmute,
    simd::{Simd, SimdFloat, SimdOrd, StdFloat},
};

// #[inline]
pub fn fast_exp(output: &mut [f32], input: &[f32]) {
    const LOWER_RANGE: f32 = -103.9720840454f32;
    const UPPER_RANGE: f32 = 88.7762626647950f32;
    const ROUNDING_BIAS: f32 = 12582912.0f32;
    const LOG2RECIPROCAL: f32 = 1.44269504088896341f32;
    const LOG2HIGH: f32 = -6.93145752e-1f32;
    const LOG2LOW: f32 = -1.42860677e-6f32;
    const POLY_0: f32 = 0.0013780593872f32;
    const POLY_1: f32 = 0.0083731245250f32;
    const POLY_2: f32 = 0.0416695363820f32;
    const POLY_3: f32 = 0.1666647195816f32;
    const POLY_4: f32 = 0.4999998509884f32;
    const POLY_56: f32 = 1.0000000000000f32;
    const MINIMUM_EXPONENT: i32 = -1056964608i32;
    const MAXIMUM_EXPONENT: i32 = 0x3F800000i32;

    const SIMD_LEN: usize = 8;

    assert_eq!(input.len(), output.len());

    let mut input = input;
    let mut output = output;
    let mut len = output.len();

    while len >= SIMD_LEN {
        let vals = Simd::<_, SIMD_LEN>::from_slice(input);
        let vals = vals.simd_clamp(Simd::splat(LOWER_RANGE), Simd::splat(UPPER_RANGE));

        let biased = vals.mul_add(Simd::splat(LOG2RECIPROCAL), Simd::splat(ROUNDING_BIAS));
        let m = biased - Simd::splat(ROUNDING_BIAS);

        let vals = m.mul_add(Simd::splat(LOG2HIGH), vals);
        let vals = m.mul_add(Simd::splat(LOG2LOW), vals);

        let overflow = unsafe { transmute::<_, Simd<i32, SIMD_LEN>>(biased) } << Simd::splat(23);
        let normal = overflow.simd_min(Simd::splat(MAXIMUM_EXPONENT));
        let normal = normal.simd_max(Simd::splat(MINIMUM_EXPONENT));
        let overflow = overflow - normal;
        let overflow = overflow + Simd::splat(MAXIMUM_EXPONENT);
        let normal = normal + Simd::splat(MAXIMUM_EXPONENT);

        let p = Simd::splat(POLY_0);
        let p = p.mul_add(vals, Simd::splat(POLY_1));
        let p = p.mul_add(vals, Simd::splat(POLY_2));
        let p = p.mul_add(vals, Simd::splat(POLY_3));
        let p = p.mul_add(vals, Simd::splat(POLY_4));
        let p = p.mul_add(vals, Simd::splat(POLY_56));

        let vals = vals * unsafe { transmute::<_, Simd<f32, SIMD_LEN>>(overflow) };
        let p = p.mul_add(vals, unsafe {
            transmute::<_, Simd<f32, SIMD_LEN>>(overflow)
        });
        let p = p * unsafe { transmute::<_, Simd<f32, SIMD_LEN>>(normal) };
        output[0..SIMD_LEN].copy_from_slice(p.as_ref());

        (input, output) = (&input[SIMD_LEN..], &mut output[SIMD_LEN..]);
        len -= SIMD_LEN
    }

    for (&val, out) in input.iter().zip(output.iter_mut()) {
        let val = val.clamp(LOWER_RANGE, UPPER_RANGE);

        let biased = val.mul_add(LOG2RECIPROCAL, ROUNDING_BIAS);
        let m = biased - ROUNDING_BIAS;

        let val = m.mul_add(LOG2HIGH, val);
        let val = m.mul_add(LOG2LOW, val);

        let overflow = unsafe { transmute::<_, i32>(biased) } << 23i32;
        let normal = overflow.min(MAXIMUM_EXPONENT);
        let normal = normal.max(MINIMUM_EXPONENT);
        let overflow = overflow - normal;
        let overflow = overflow + MAXIMUM_EXPONENT;
        let normal = normal + MAXIMUM_EXPONENT;

        let p = POLY_0;
        let p = p.mul_add(val, POLY_1);
        let p = p.mul_add(val, POLY_2);
        let p = p.mul_add(val, POLY_3);
        let p = p.mul_add(val, POLY_4);
        let p = p.mul_add(val, POLY_56);

        let val = val * unsafe { transmute::<_, f32>(overflow) };
        let p = p.mul_add(val, unsafe { transmute::<_, f32>(overflow) });
        let p = p * unsafe { transmute::<_, f32>(normal) };

        *out = p;
    }
}
