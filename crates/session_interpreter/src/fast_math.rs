use std::{
    mem::transmute,
    simd::{
        num::SimdUint,
        prelude::{Simd, SimdFloat, SimdOrd},
        StdFloat,
    },
};

// NOTE: No longer used in softmax kernel, but will be used for exp kernel in the future.
// #[allow(dead_code)]
// pub fn fast_exp(output: &mut [f32], input: &[f32]) {
//     const LOWER_RANGE: f32 = -103.9720840454f32;
//     const UPPER_RANGE: f32 = 88.7762626647950f32;
//     const ROUNDING_BIAS: f32 = 12582912.0f32;
//     const LOG2RECIPROCAL: f32 = 1.44269504088896341f32;
//     const LOG2HIGH: f32 = -6.93145752e-1f32;
//     const LOG2LOW: f32 = -1.42860677e-6f32;
//     const POLY_0: f32 = 0.0013780593872f32;
//     const POLY_1: f32 = 0.0083731245250f32;
//     const POLY_2: f32 = 0.0416695363820f32;
//     const POLY_3: f32 = 0.1666647195816f32;
//     const POLY_4: f32 = 0.4999998509884f32;
//     const POLY_56: f32 = 1.0000000000000f32;
//     const MINIMUM_EXPONENT: i32 = -1056964608i32;
//     const MAXIMUM_EXPONENT: i32 = 0x3F800000i32;
//
//     const SIMD_LEN: usize = 8;
//
//     assert_eq!(input.len(), output.len());
//
//     let mut input = input;
//     let mut output = output;
//     let mut len = output.len();
//
//     while len >= SIMD_LEN {
//         let vals = Simd::<_, SIMD_LEN>::from_slice(input);
//         let vals = vals.simd_clamp(Simd::splat(LOWER_RANGE), Simd::splat(UPPER_RANGE));
//
//         let biased = vals.mul_add(Simd::splat(LOG2RECIPROCAL), Simd::splat(ROUNDING_BIAS));
//         let m = biased - Simd::splat(ROUNDING_BIAS);
//
//         let vals = m.mul_add(Simd::splat(LOG2HIGH), vals);
//         let vals = m.mul_add(Simd::splat(LOG2LOW), vals);
//
//         let overflow = unsafe { transmute::<_, Simd<i32, SIMD_LEN>>(biased) } << Simd::splat(23);
//         let normal = overflow.simd_min(Simd::splat(MAXIMUM_EXPONENT));
//         let normal = normal.simd_max(Simd::splat(MINIMUM_EXPONENT));
//         let overflow = overflow - normal;
//         let overflow = overflow + Simd::splat(MAXIMUM_EXPONENT);
//         let normal = normal + Simd::splat(MAXIMUM_EXPONENT);
//
//         let p = Simd::splat(POLY_0);
//         let p = p.mul_add(vals, Simd::splat(POLY_1));
//         let p = p.mul_add(vals, Simd::splat(POLY_2));
//         let p = p.mul_add(vals, Simd::splat(POLY_3));
//         let p = p.mul_add(vals, Simd::splat(POLY_4));
//         let p = p.mul_add(vals, Simd::splat(POLY_56));
//
//         let vals = vals * unsafe { transmute::<_, Simd<f32, SIMD_LEN>>(overflow) };
//         let p = p.mul_add(vals, unsafe {
//             transmute::<_, Simd<f32, SIMD_LEN>>(overflow)
//         });
//         let p = p * unsafe { transmute::<_, Simd<f32, SIMD_LEN>>(normal) };
//         output[0..SIMD_LEN].copy_from_slice(p.as_ref());
//
//         (input, output) = (&input[SIMD_LEN..], &mut output[SIMD_LEN..]);
//         len -= SIMD_LEN
//     }
//
//     for (&val, out) in input.iter().zip(output.iter_mut()) {
//         let val = val.clamp(LOWER_RANGE, UPPER_RANGE);
//
//         let biased = val.mul_add(LOG2RECIPROCAL, ROUNDING_BIAS);
//         let m = biased - ROUNDING_BIAS;
//
//         let val = m.mul_add(LOG2HIGH, val);
//         let val = m.mul_add(LOG2LOW, val);
//
//         let overflow = (biased.to_bits() as i32) << 23i32;
//         let normal = overflow.min(MAXIMUM_EXPONENT);
//         let normal = normal.max(MINIMUM_EXPONENT);
//         let overflow = overflow - normal;
//         let overflow = overflow + MAXIMUM_EXPONENT;
//         let normal = normal + MAXIMUM_EXPONENT;
//
//         let p = POLY_0;
//         let p = p.mul_add(val, POLY_1);
//         let p = p.mul_add(val, POLY_2);
//         let p = p.mul_add(val, POLY_3);
//         let p = p.mul_add(val, POLY_4);
//         let p = p.mul_add(val, POLY_56);
//
//         let val = val * f32::from_bits(overflow as u32);
//         let p = p.mul_add(val, f32::from_bits(overflow as u32));
//         let p = p * f32::from_bits(normal as u32);
//
//         *out = p;
//     }
// }

pub fn fast_sum_exp(output: &mut [f32], input: &[f32]) -> f32 {
    // const LOWER_RANGE: f32 = -103.9720840454f32;
    // const UPPER_RANGE: f32 = 88.7762626647950f32;
    const LOWER_RANGE_2: f32 = -88.37626f32;
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
    // const MINIMUM_EXPONENT: i32 = -1056964608i32;
    const MAXIMUM_EXPONENT: i32 = 0x3F800000i32;

    const SIMD_LEN: usize = 8;

    assert_eq!(input.len(), output.len());

    let mut input = input;
    let mut output = output;
    let mut len = output.len();
    let mut sum = Simd::<f32, SIMD_LEN>::splat(0f32);
    let max = input.iter().fold(f32::NAN, |m, v| v.max(m));

    while len >= SIMD_LEN {
        let vals = Simd::<_, SIMD_LEN>::from_slice(input) - Simd::splat(max);
        let vals = vals.simd_max(Simd::splat(LOWER_RANGE_2));

        let biased = vals.mul_add(Simd::splat(LOG2RECIPROCAL), Simd::splat(ROUNDING_BIAS));
        let m = biased - Simd::splat(ROUNDING_BIAS);

        let vals = m.mul_add(Simd::splat(LOG2HIGH), vals);
        let vals = m.mul_add(Simd::splat(LOG2LOW), vals);

        let normal = biased.to_bits().cast::<i32>() << Simd::splat(23);
        let normal = normal + Simd::splat(MAXIMUM_EXPONENT);

        let p = Simd::splat(POLY_0);
        let p = p.mul_add(vals, Simd::splat(POLY_1));
        let p = p.mul_add(vals, Simd::splat(POLY_2));
        let p = p.mul_add(vals, Simd::splat(POLY_3));
        let p = p.mul_add(vals, Simd::splat(POLY_4));
        let p = p.mul_add(vals, Simd::splat(POLY_56));
        let p = p.mul_add(vals, Simd::splat(POLY_56));

        let p = p * unsafe { transmute::<_, Simd<f32, SIMD_LEN>>(normal) };
        sum += p;
        output[0..SIMD_LEN].copy_from_slice(p.as_ref());

        (input, output) = (&input[SIMD_LEN..], &mut output[SIMD_LEN..]);
        len -= SIMD_LEN
    }

    let mut sum = sum.reduce_sum();

    for (&val, out) in input.iter().zip(output.iter_mut()) {
        let val = (val - max).max(LOWER_RANGE_2);

        let biased = val.mul_add(LOG2RECIPROCAL, ROUNDING_BIAS);
        let m = biased - ROUNDING_BIAS;

        let val = m.mul_add(LOG2HIGH, val);
        let val = m.mul_add(LOG2LOW, val);

        let normal = (biased.to_bits() as i32) << 23i32;
        let normal = normal + MAXIMUM_EXPONENT;

        let p = POLY_0;
        let p = p.mul_add(val, POLY_1);
        let p = p.mul_add(val, POLY_2);
        let p = p.mul_add(val, POLY_3);
        let p = p.mul_add(val, POLY_4);
        let p = p.mul_add(val, POLY_56);
        let p = p.mul_add(val, POLY_56);

        let p = p * f32::from_bits(normal as u32);
        sum += p;

        *out = p;
    }

    sum
}

pub fn fast_sigmoid(output: &mut [f32], input: &[f32]) {
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
        let vals = -Simd::<_, SIMD_LEN>::from_slice(input);
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
        output[0..SIMD_LEN].copy_from_slice((Simd::splat(1.) / (Simd::splat(1.) + p)).as_ref());

        (input, output) = (&input[SIMD_LEN..], &mut output[SIMD_LEN..]);
        len -= SIMD_LEN
    }

    for (&val, out) in input.iter().zip(output.iter_mut()) {
        let val = (-val).clamp(LOWER_RANGE, UPPER_RANGE);

        let biased = val.mul_add(LOG2RECIPROCAL, ROUNDING_BIAS);
        let m = biased - ROUNDING_BIAS;

        let val = m.mul_add(LOG2HIGH, val);
        let val = m.mul_add(LOG2LOW, val);

        let overflow = (biased.to_bits() as i32) << 23i32;
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

        let val = val * f32::from_bits(overflow as u32);
        let p = p.mul_add(val, f32::from_bits(overflow as u32));
        let p = p * f32::from_bits(normal as u32);

        *out = 1. / (1. + p);
    }
}

pub fn fast_gelu(mut output: &mut [f32], mut input: &[f32]) {
    const B: f32 = 0.7978845608028654f32; // sqrt(2.0 / PI)
    const C: f32 = 0.035677408136300125f32; // 0.044715 * sqrt(2.0 / PI)
    const LOWER_RANGE: f32 = -9f32;
    const UPPER_RANGE: f32 = 9f32;
    const ALPHA_13: f32 = -2.76076847742355e-16f32;
    const ALPHA_11: f32 = 2.00018790482477e-13f32;
    const ALPHA_9: f32 = -8.60467152213735e-11f32;
    const ALPHA_7: f32 = 5.12229709037114e-08f32;
    const ALPHA_5: f32 = 1.48572235717979e-05f32;
    const ALPHA_3: f32 = 6.37261928875436e-04f32;
    const ALPHA_1: f32 = 4.89352455891786e-03f32;
    const BETA_6: f32 = 1.19825839466702e-06f32;
    const BETA_4: f32 = 1.18534705686654e-04f32;
    const BETA_2: f32 = 2.26843463243900e-03f32;
    const BETA_0: f32 = 4.89352518554385e-03f32;

    const SIMD_LEN: usize = 8;

    assert_eq!(output.len(), input.len());
    let mut len = output.len();

    while len >= SIMD_LEN {
        let i = Simd::<f32, SIMD_LEN>::from_slice(input);
        let vals = i * (Simd::splat(C) * i * i + Simd::splat(B));
        let vals = vals.simd_clamp(Simd::splat(LOWER_RANGE), Simd::splat(UPPER_RANGE));

        let vals_squared = vals * vals;

        let p = vals_squared.mul_add(Simd::splat(ALPHA_13), Simd::splat(ALPHA_11));
        let p = p.mul_add(vals_squared, Simd::splat(ALPHA_9));
        let p = p.mul_add(vals_squared, Simd::splat(ALPHA_7));
        let p = p.mul_add(vals_squared, Simd::splat(ALPHA_5));
        let p = p.mul_add(vals_squared, Simd::splat(ALPHA_3));
        let p = p.mul_add(vals_squared, Simd::splat(ALPHA_1));
        let p = p * vals;

        let q = vals_squared.mul_add(Simd::splat(BETA_6), Simd::splat(BETA_4));
        let q = q.mul_add(vals_squared, Simd::splat(BETA_2));
        let q = q.mul_add(vals_squared, Simd::splat(BETA_0));

        let o = p / q;
        let o = (o + Simd::splat(1.)) * (i * Simd::splat(0.5));

        output[0..SIMD_LEN].copy_from_slice(o.as_ref());

        len -= SIMD_LEN;
        input = &input[SIMD_LEN..];
        output = &mut output[SIMD_LEN..];
    }

    for (&i, out) in input.iter().zip(output.iter_mut()) {
        let val = i * (C * i * i + B);
        let val = val.clamp(LOWER_RANGE, UPPER_RANGE);

        let val_squared = val * val;

        let p = val_squared.mul_add(ALPHA_13, ALPHA_11);
        let p = p.mul_add(val_squared, ALPHA_9);
        let p = p.mul_add(val_squared, ALPHA_7);
        let p = p.mul_add(val_squared, ALPHA_5);
        let p = p.mul_add(val_squared, ALPHA_3);
        let p = p.mul_add(val_squared, ALPHA_1);
        let p = p * val;

        let q = val_squared.mul_add(BETA_6, BETA_4);
        let q = q.mul_add(val_squared, BETA_2);
        let q = q.mul_add(val_squared, BETA_0);

        let o = p / q;
        let o = (o + 1.) * (i * 0.5);

        *out = o
    }
}
