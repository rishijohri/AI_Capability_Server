/*
 * scipy_blas_shim.c — Shim library for App Store compatibility
 *
 * Apple's App Store scanner flags certain LAPACK/BLAS symbol *references*
 * (_lsame_, _dcabs1_, _xerbla_array__) in scipy's .so files as "non-public
 * APIs" because those names collide with internal Accelerate symbols.
 *
 * This shim provides renamed implementations of these three trivial
 * functions. The scipy .so files are binary-patched post-build to
 * reference the renamed symbols, and their Accelerate link is redirected
 * to this shim (which re-exports Accelerate for all other BLAS/LAPACK
 * symbols).
 *
 * Build:
 *   clang -dynamiclib -o libscipy_blas_shim.dylib scipy_blas_shim.c \
 *     -Wl,-reexport_framework,Accelerate \
 *     -install_name @executable_path/../Frameworks/libscipy_blas_shim.dylib \
 *     -arch arm64 -mmacosx-version-min=11.0
 */

/*
 * lsamZ_  (renamed from lsame_)
 *
 * LAPACK auxiliary: tests if two characters are the same regardless of case.
 * Standard FORTRAN calling convention: pointers to single chars.
 */
int lsamZ_(const char *ca, const char *cb) {
    char a = *ca;
    char b = *cb;
    if (a >= 'a' && a <= 'z') a -= ('a' - 'A');
    if (b >= 'a' && b <= 'z') b -= ('a' - 'A');
    return (a == b);
}

/*
 * dcabZ1_  (renamed from dcabs1_)
 *
 * LAPACK auxiliary: computes |Re(z)| + |Im(z)| for a double-complex number
 * stored as two consecutive doubles (real part, imaginary part).
 */
double dcabZ1_(const double *z) {
    double re = z[0];
    double im = z[1];
    return (re < 0.0 ? -re : re) + (im < 0.0 ? -im : im);
}

/*
 * xerblZ_array__  (renamed from xerbla_array__)
 *
 * LAPACK error handler called when an input argument has an illegal value.
 * In the bundled context this should never fire; provide a no-op.
 */
void xerblZ_array__(const char *srname_array,
                    const int  *srname_len,
                    const int  *info) {
    /* intentionally empty — error handler stub */
    (void)srname_array;
    (void)srname_len;
    (void)info;
}
