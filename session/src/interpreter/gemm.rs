pub fn sgemm(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    unsafe {
        #[cfg(not(feature = "cblas"))]
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            lda as isize,
            1,
            b.as_ptr(),
            ldb as isize,
            1,
            beta,
            c.as_mut_ptr(),
            ldc as isize,
            1,
        );
        #[cfg(feature = "cblas")]
        {
            cblas_sys::cblas_sgemm(
                cblas_sys::CblasRowMajor,
                cblas_sys::CblasNoTrans,
                cblas_sys::CblasNoTrans,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
    }
}
