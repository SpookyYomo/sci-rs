use core::{f64::consts::PI, iter::Sum, ops::Mul};

use nalgebra::{Complex, ComplexField, RealField};
use num_traits::Float;

#[cfg(feature = "alloc")]
use super::{
    bilinear_zpk_dyn, lp2bp_zpk_dyn, lp2bs_zpk_dyn, lp2hp_zpk_dyn, lp2lp_zpk_dyn, zpk2sos_dyn,
    DigitalFilter, FilterBandType, FilterOutputType, FilterType, Sos,
};
#[cfg(feature = "alloc")]
use crate::signal::filter::design::{zpk2tf_dyn, ZpkFormatFilter};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

///
///
/// IIR digital and analog filter design given order and critical points.
///
/// Design an Nth-order digital or analog filter and return the filter
/// coefficients.
///
/// -------
/// b, a : ndarray, ndarray
///     Numerator ('b') and denominator ('a') polynomials of the IIR filter.
///     Only returned if 'output='ba'.
/// z, p, k : ndarray, ndarray, float
///     Zeros, poles, and system gain of the IIR filter transfer
///     function.  Only returned if 'output='zpk'.
/// sos : ndarray
///     Second-order sections representation of the IIR filter.
///     Only returned if 'output=='sos'.
///
/// See Also
/// --------
/// butter : Filter design using order and critical points
/// cheby1, cheby2, ellip, bessel
/// buttord : Find order and critical points from passband and stopband spec
/// cheb1ord, cheb2ord, ellipord
/// iirdesign : General filter design using passband and stopband spec
///
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "alloc")]
pub fn iirfilter_dyn<F>(
    order: usize,
    wn: Vec<F>,
    rp: Option<F>,
    rs: Option<F>,
    btype: Option<FilterBandType>,
    ftype: Option<FilterType>,
    analog: Option<bool>,
    output: Option<FilterOutputType>,
    fs: Option<F>,
) -> DigitalFilter<F>
where
    F: RealField + Float + Sum,
{
    use super::bilinear_zpk_dyn;

    let analog = analog.unwrap_or(false);
    let mut wn = wn;

    if wn.len() > 2 {
        panic!("Wn may be of len 1 or 2");
    }

    if let Some(fs) = fs {
        if analog {
            panic!("fs cannot be specified for an analog filter");
        }

        wn.iter_mut().for_each(|wni| {
            *wni = F::from(2.).unwrap() * *wni / fs;
        });
    }

    if wn.iter().any(|wi| *wi <= F::zero()) {
        panic!("filter critical frequencies must be greater than 0");
    }

    if wn.len() > 1 && wn[0] >= wn[1] {
        panic!("Wn[0] must be less than Wn[1]");
    }

    if let Some(rp) = rp {
        if rp < F::zero() {
            panic!("passband ripple (rp) must be positive");
        }
    }

    if let Some(rs) = rs {
        if rs < F::zero() {
            panic!("stopband attenuation (rs) must be positive");
        }
    }

    // Get analog lowpass prototype
    let ftype = ftype.unwrap_or(FilterType::Butterworth);
    let zpk: ZpkFormatFilter<F> = match ftype {
        FilterType::Butterworth => buttap_dyn(order),
        FilterType::ChebyshevI => {
            if rp.is_none() {
                panic!("passband ripple (rp) must be provided to design a Chebyshev I filter");
            }
            cheb1ap_dyn(order, rp.unwrap())
        }
        FilterType::ChebyshevII => {
            if rp.is_none() {
                panic!(
                    "stopband attenuation (rs) must be provided to design an Chebyshev II filter."
                );
            }
            cheb2ap_dyn(order, rs.unwrap())
        }
        FilterType::CauerElliptic => {
            if rs.is_none() || rp.is_none() {
                panic!("Both rp and rs must be provided to design an elliptic filter.");
            }
            // ellipap::<N>(rp, rs)
            todo!()
        }
        FilterType::BesselThomson(norm) => {
            // besselap::<N>(norm = norm),
            todo!()
        }
    };

    // Pre-warp frequencies for digital filter design
    let (fs, warped) = if !analog {
        if wn.iter().any(|wi| *wi <= F::zero() || *wi >= F::one()) {
            if let Some(fs) = fs {
                panic!(
                    "Digital filter critical frequencies must be 0 < Wn < fs/2 (fs={} -> fs/2={})",
                    fs,
                    fs / F::from(2.).unwrap()
                );
            }
            panic!("Digital filter critical frequencies must be 0 < Wn < 1");
        }
        let fs = F::from(2.).unwrap();
        let mut warped = wn
            .iter()
            .map(|wni| F::from(2.).unwrap() * fs * Float::tan(F::from(PI).unwrap() * *wni / fs))
            .collect::<Vec<_>>();
        (fs, warped)
    } else {
        (fs.unwrap_or_else(F::one), wn.clone())
    };

    // transform to lowpass, bandpass, highpass, or bandstop
    let btype = btype.unwrap_or(FilterBandType::Bandpass);
    let zpk = match btype {
        FilterBandType::Lowpass => {
            if wn.len() != 1 {
                panic!(
                    "Must specify a single critical frequency Wn for lowpass or highpass filter"
                );
            }

            lp2lp_zpk_dyn(zpk, Some(warped[0]))
        }
        FilterBandType::Highpass => {
            if wn.len() != 1 {
                panic!(
                    "Must specify a single critical frequency Wn for lowpass or highpass filter"
                );
            }
            lp2hp_zpk_dyn(zpk, Some(warped[0]))
        }
        FilterBandType::Bandpass => {
            if wn.len() != 2 {
                panic!(
                    "Wn must specify start and stop frequencies for bandpass or bandstop filter"
                );
            }

            let bw = warped[1] - warped[0];
            let wo = Float::sqrt(warped[0] * warped[1]);
            lp2bp_zpk_dyn(zpk, Some(wo), Some(bw))
        }
        FilterBandType::Bandstop => {
            if wn.len() != 2 {
                panic!(
                    "Wn must specify start and stop frequencies for bandpass or bandstop filter"
                );
            }

            let bw = warped[1] - warped[0];
            let wo = Float::sqrt(warped[0] * warped[1]);
            lp2bs_zpk_dyn(zpk, Some(wo), Some(bw))
        }
    };

    // Find discrete equivalent if necessary
    let zpk = if !analog {
        bilinear_zpk_dyn(zpk, fs)
    } else {
        zpk
    };

    // Transform to proper out type (pole-zero, state-space, numer-denom)
    let output = output.unwrap_or(FilterOutputType::Ba);
    match output {
        FilterOutputType::Zpk => DigitalFilter::Zpk(zpk),
        FilterOutputType::Ba => DigitalFilter::Ba(zpk2tf_dyn(2 * order, &zpk.z, &zpk.p, zpk.k)),
        FilterOutputType::Sos => DigitalFilter::Sos(zpk2sos_dyn(order, zpk, None, Some(analog))),
    }
}

/// """Return (z,p,k) for analog prototype of Nth-order Butterworth filter.
///
/// The filter will have an angular (e.g., rad/s) cutoff frequency of 1.
///
/// See Also
/// --------
/// butter : Filter design function using this prototype
///
/// """
#[cfg(feature = "alloc")]
fn buttap_dyn<F>(order: usize) -> ZpkFormatFilter<F>
where
    F: Float + RealField + Mul<Output = F>,
{
    let p: Vec<Complex<F>> = ((-(order as isize) + 1)..order as isize)
        .step_by(2)
        .map(|i| {
            let mi = F::from(i).unwrap();
            let num = unsafe { F::from(PI).unwrap_unchecked() * mi };
            let denom = unsafe { F::from(2. * order as f64).unwrap_unchecked() };
            let c = Complex::new(F::zero(), num / denom);
            -c.exp()
        })
        .collect::<Vec<_>>();

    ZpkFormatFilter::new(Vec::new(), p, F::one())
}

/// Return (z,p,k) for Nth-order Chebyshev type I analog lowpass filter.
///
/// The returned filter prototype has `rp` decibels of ripple in the passband.
///
/// The filter's angular (e.g. rad/s) cutoff frequency is normalized to 1,
/// defined as the point at which the gain first drops below ``-rp``.
///
/// See Also
/// --------
/// cheby1 : Filter design function using this prototype
#[cfg(feature = "alloc")]
pub fn cheb1ap_dyn<F>(n: usize, rp: F) -> ZpkFormatFilter<F>
where
    F: Float + RealField,
{
    let ten = F::from(10).unwrap();
    if n == 0 {
        return ZpkFormatFilter {
            z: Vec::new(),
            p: Vec::new(),
            k: Float::powf(ten, (-rp / F::from(20).unwrap())),
        };
    }
    let eps = Float::sqrt(Float::powf(ten, rp / ten) - F::one());
    let p: Vec<Complex<F>> = {
        let mu = Float::asinh(F::one() / eps) / F::from(n).unwrap();
        let n = n as isize;
        let ms = ((-n + 1)..n).step_by(2);
        let thetas = ms.map(|m| F::pi() * (F::from(m).unwrap() / F::from(2 * n).unwrap()));

        thetas
            .map(|theta| -Complex::sinh(Complex::new(mu, theta)))
            .collect()
    };
    let mut k = p
        .iter()
        .map(|z| -(*z))
        .fold(Complex::new(F::one(), F::zero()), |acc, z| acc * z)
        .real();
    if n % 2 == 0 {
        k /= Float::sqrt(F::one() + eps * eps);
    }

    let z = Vec::new();
    ZpkFormatFilter { z, p, k }
}

/// Chebyshev type I digital and analog filter design.
///
/// Design an Nth-order digital or analog Chebyshev type I filter and
/// return the filter coefficients.
///
/// Parameters
/// ----------
/// * `N` : int  
///   The order of the filter.
/// * `rp` : float  
///   The maximum ripple allowed below unity gain in the passband.
///   Specified in decibels, as a positive number.
/// * `Wn` : array_like  
///   A scalar or length-2 sequence giving the critical frequencies.
///   For Type I filters, this is the point in the transition band at which
///   the gain first drops below -`rp`.
///
///   For digital filters, `Wn` are in the same units as `fs`. By default,
///   `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
///   where 1 is the Nyquist frequency. (`Wn` is thus in
///   half-cycles / sample.)
///
///   For analog filters, `Wn` is an angular frequency (e.g., rad/s).
/// * `btype` : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional  
///   The type of filter.  Default is 'lowpass'.
/// * `analog` : bool, optional  
///   When True, return an analog filter, otherwise a digital filter is
///   returned.
/// * `output` : {'ba', 'zpk', 'sos'}, optional  
///   Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
///   second-order sections ('sos'). Default is 'ba' for backwards
///   compatibility, but 'sos' should be used for general-purpose filtering.
/// * `fs` : float, optional  
///   The sampling frequency of the digital system.
///
/// Returns
/// -------
/// b, a : ndarray, ndarray  
///     Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
///     Only returned if ``output='ba'``.
/// z, p, k : ndarray, ndarray, float  
///     Zeros, poles, and system gain of the IIR filter transfer
///     function.  Only returned if ``output='zpk'``.
/// sos : ndarray  
///     Second-order sections representation of the IIR filter.
///     Only returned if ``output='sos'``.
///
/// See Also
/// --------
/// cheb1ord, [cheb1ap_dyn]
///
/// Notes
/// -----
/// The Chebyshev type I filter maximizes the rate of cutoff between the
/// frequency response's passband and stopband, at the expense of ripple in
/// the passband and increased ringing in the step response.
///
/// Type I filters roll off faster than Type II (`cheby2`), but Type II
/// filters do not have any ripple in the passband.
///
/// The equiripple passband has N maxima or minima (for example, a
/// 5th-order filter has 3 maxima and 2 minima). Consequently, the DC gain is
/// unity for odd-order filters, or -rp dB for even-order filters.
#[cfg(feature = "alloc")]
pub fn cheby1_dyn<F>(
    n: usize,
    rp: F,
    wn: Vec<F>,
    btype: Option<FilterBandType>,
    analog: Option<bool>,
    output: Option<FilterOutputType>,
    fs: Option<F>,
) -> DigitalFilter<F>
where
    F: RealField + Float + Sum,
{
    iirfilter_dyn(
        n,
        wn,
        Some(rp),
        None, // rs
        btype,
        Some(FilterType::ChebyshevI),
        analog,
        output,
        fs,
    )
}

/// Return (z,p,k) for Nth-order Chebyshev type II analog lowpass filter.
///
/// The returned filter prototype has attenuation of at least ``rs`` decibels
/// in the stopband.
///
/// The filter's angular (e.g. rad/s) cutoff frequency is normalized to 1,
/// defined as the point at which the attenuation first reaches ``rs``.
///
/// See Also
/// --------
/// cheby2 : Filter design function using this prototype
#[cfg(feature = "alloc")]
pub fn cheb2ap_dyn<F>(n: usize, rs: F) -> ZpkFormatFilter<F>
where
    F: Float + RealField,
{
    if n == 0 {
        return ZpkFormatFilter {
            z: Vec::new(),
            p: Vec::new(),
            k: F::one(),
        };
    }

    let ten = F::from(10).unwrap();
    // Ripple factor
    let de = F::one() / Float::sqrt(Float::powf(ten, (F::one() / ten) * rs) - F::one());
    let mu = Float::asinh(F::one() / de) / F::from(n).unwrap();

    let m = if n % 2 == 1 {
        let n = n as isize;
        ((-n + 1)..0).step_by(2).chain((2..n).step_by(2))
    } else {
        let n = n as isize;
        ((-n + 1)..n).step_by(2).chain((0..0).step_by(2))
    };
    let imag: Complex<F> = Complex::i();
    let two = F::from(2).unwrap();
    let c_unit = Complex::new(F::one(), F::zero());
    let z: Vec<Complex<F>> = {
        let n = F::from(n).unwrap();
        m.map(|m| F::from(m).unwrap() * (F::pi() / (two * n)))
            .map(|x| Float::sin(x))
            .map(|x| -(imag / Complex::new(x, F::zero())).conj())
            .collect()
    };
    let p: Vec<_> = {
        let n = n as isize;
        ((-n + 1)..n)
            .step_by(2)
            .map(|x| {
                Complex::new(
                    F::zero(),
                    F::pi() * F::from(x).unwrap() / (two * F::from(n).unwrap()),
                )
            })
            .map(|z| -Complex::exp(z))
            .map(|z| Complex::new(Float::sinh(mu) * z.real(), Float::cosh(mu) * z.imaginary()))
            .map(|z| c_unit / z)
            .collect()
    };
    let k = (p.iter().map(|x| -x).fold(c_unit, |acc, x| acc * x)
        / z.iter().map(|x| -x).fold(c_unit, |acc, x| acc * x))
    .real();

    ZpkFormatFilter { z, p, k }
}

/// Chebyshev type II digital and analog filter design.
///
/// Design an Nth-order digital or analog Chebyshev type II filter and
/// return the filter coefficients.
///
/// Parameters
/// ----------
/// * `N` : int  
///   The order of the filter.
/// * `rs` : float  
///   The minimum attenuation required in the stop band.
///   Specified in decibels, as a positive number.
/// * `Wn` : array_like  
///   A scalar or length-2 sequence giving the critical frequencies.
///   For Type II filters, this is the point in the transition band at which
///   the gain first reaches -`rs`.
///
///   For digital filters, `Wn` are in the same units as `fs`. By default,
///   `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
///   where 1 is the Nyquist frequency. (`Wn` is thus in
///   half-cycles / sample.)
///
///   For analog filters, `Wn` is an angular frequency (e.g., rad/s).
/// * `btype` : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional  
///   The type of filter.  Default is 'lowpass'.
/// * `analog` : bool, optional  
///   When True, return an analog filter, otherwise a digital filter is
///   returned.
/// * `output` : {'ba', 'zpk', 'sos'}, optional  
///   Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
///   second-order sections ('sos'). Default is 'ba' for backwards
///   compatibility, but 'sos' should be used for general-purpose filtering.
/// * `fs` : float, optional  
///   The sampling frequency of the digital system.
///
/// Returns
/// -------
/// b, a : ndarray, ndarray
///   Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
///   Only returned if ``output='ba'``.
/// z, p, k : ndarray, ndarray, float
///   Zeros, poles, and system gain of the IIR filter transfer
///   function.  Only returned if ``output='zpk'``.
/// sos : ndarray
///   Second-order sections representation of the IIR filter.
///   Only returned if ``output='sos'``.
///
/// See Also
/// --------
/// cheb2ord, [cheb2ap_dyn]
///
/// Notes
/// -----
/// The Chebyshev type II filter maximizes the rate of cutoff between the
/// frequency response's passband and stopband, at the expense of ripple in
/// the stopband and increased ringing in the step response.
///
/// Type II filters do not roll off as fast as Type I (`cheby1`).
#[cfg(feature = "alloc")]
pub fn cheby2_dyn<F>(
    n: usize,
    rs: F,
    wn: Vec<F>,
    btype: Option<FilterBandType>,
    analog: Option<bool>,
    output: Option<FilterOutputType>,
    fs: Option<F>,
) -> DigitalFilter<F>
where
    F: RealField + Float + Sum,
{
    iirfilter_dyn(
        n,
        wn,
        None, // rp
        Some(rs),
        btype,
        Some(FilterType::ChebyshevII),
        analog,
        output,
        fs,
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn matches_scipy_buttap() {
        let p: [Complex<f64>; 4] = [
            Complex::new(-0.38268343, 0.92387953),
            Complex::new(-0.92387953, 0.38268343),
            Complex::new(-0.92387953, -0.38268343),
            Complex::new(-0.38268343, -0.92387953),
        ];
        let zpk = buttap_dyn::<f64>(4);
        for (expected, actual) in p.into_iter().zip(zpk.p) {
            assert_relative_eq!(expected.re, actual.re, max_relative = 1e-7);
            assert_relative_eq!(expected.im, actual.im, max_relative = 1e-7);
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn matches_scipy_cheb1ap() {
        {
            // from scipy.signal import cheb1ap
            // cheb1ap(N=4, rp=2) = (array([], dtype=float64), array(
            //    [-0.10488725+0.95795296j,
            //     -0.25322023+0.39679711j,
            //     -0.25322023-0.39679711j,
            //     -0.10488725-0.95795296j]),
            //   np.float64(0.1634450339473848))
            let p: [Complex<f64>; 4] = [
                Complex::new(-0.10488725, 0.95795296),
                Complex::new(-0.25322023, 0.39679711),
                Complex::new(-0.25322023, -0.39679711),
                Complex::new(-0.10488725, -0.95795296),
            ];
            let k = 0.1634450339473848;

            let zpk = cheb1ap_dyn::<f64>(4, 2.);
            for (expected, actual) in p.into_iter().zip(zpk.p) {
                assert_relative_eq!(expected.re, actual.re, max_relative = 1e-7);
                assert_relative_eq!(expected.im, actual.im, max_relative = 1e-7);
            }
            assert_relative_eq!(zpk.k, k);
        }
        {
            // from scipy.signal import cheb1ap
            // cheb1ap(N=5, rp=2) = (array([], dtype=float64), array(
            //    [-0.06746098+0.97345572j,
            //     -0.17661514+0.60162872j,
            //     -0.21830832-0.j        ,
            //     -0.17661514-0.60162872j,
            //     -0.06746098-0.97345572j]),
            //   np.float64(0.08172251697369243))
            let p: [Complex<f64>; 5] = [
                Complex::new(-0.06746098, 0.97345572),
                Complex::new(-0.17661514, 0.60162872),
                Complex::new(-0.21830832, -0.),
                Complex::new(-0.17661514, -0.60162872),
                Complex::new(-0.06746098, -0.97345572),
            ];
            let k = 0.08172251697369243;

            let zpk = cheb1ap_dyn::<f64>(5, 2.);
            for (expected, actual) in p.into_iter().zip(zpk.p) {
                assert_relative_eq!(expected.re, actual.re, max_relative = 1e-7);
                assert_relative_eq!(expected.im, actual.im, max_relative = 1e-7);
            }
            assert_relative_eq!(zpk.k, k);
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn matches_scipy_cheb2ap() {
        {
            // from scipy.signal import cheb2ap
            // cheb2ap(N=4, rs=2) = (
            // array([ 0.-1.0823922j ,  0.-2.61312593j,
            //        -0.+2.61312593j, -0.+1.0823922j ]),
            // array([-0.07660576-1.06026362j, -0.92034183-2.18549705j,
            //        -0.92034183+2.18549705j, -0.07660576+1.06026362j]),
            // np.float64(0.7943282347242814))
            let z: [Complex<f64>; 4] = [
                Complex::new(0., -1.0823922),
                Complex::new(0., -2.61312593),
                Complex::new(-0., 2.61312593),
                Complex::new(-0., 1.0823922),
            ];
            let p: [Complex<f64>; 4] = [
                Complex::new(-0.07660576, -1.06026362),
                Complex::new(-0.92034183, -2.18549705),
                Complex::new(-0.92034183, 2.18549705),
                Complex::new(-0.07660576, 1.06026362),
            ];
            let k = 0.7943282347242814;

            let zpk = cheb2ap_dyn::<f64>(4, 2.);
            for (expected, actual) in z.into_iter().zip(zpk.z) {
                assert_relative_eq!(expected.re, actual.re, max_relative = 1e-7);
                assert_relative_eq!(expected.im, actual.im, max_relative = 1e-7);
            }
            for (expected, actual) in p.into_iter().zip(zpk.p) {
                assert_relative_eq!(expected.re, actual.re, max_relative = 1e-7);
                assert_relative_eq!(expected.im, actual.im, max_relative = 1e-7);
            }
            assert_relative_eq!(zpk.k, k);
        }
        {
            // from scipy.signal import cheb2ap
            // cheb2ap(N=5, rs=2) = (
            // array([ 0.-1.05146222j,  0.-1.70130162j,
            //        -0.+1.70130162j, -0.+1.05146222j]),
            // array([-0.04728049-1.0389464j , -0.31310088-1.62417385j,
            //        -7.06944213-0.j        , -0.31310088+1.62417385j,
            //        -0.04728049+1.0389464j ]),
            // np.float64(6.537801357895397))
            let z: [Complex<f64>; 4] = [
                Complex::new(0., -1.05146222),
                Complex::new(0., -1.70130162),
                Complex::new(-0., 1.70130162),
                Complex::new(-0., 1.05146222),
            ];
            let p: [Complex<f64>; 5] = [
                Complex::new(-0.04728049, -1.0389464),
                Complex::new(-0.31310088, -1.62417385),
                Complex::new(-7.06944213, -0.),
                Complex::new(-0.31310088, 1.62417385),
                Complex::new(-0.04728049, 1.0389464),
            ];
            let k = 6.537801357895397;

            let zpk = cheb2ap_dyn::<f64>(5, 2.);
            for (expected, actual) in z.into_iter().zip(zpk.z) {
                assert_relative_eq!(expected.re, actual.re, max_relative = 1e-7);
                assert_relative_eq!(expected.im, actual.im, max_relative = 1e-7);
            }
            for (expected, actual) in p.into_iter().zip(zpk.p) {
                assert_relative_eq!(expected.re, actual.re, max_relative = 1e-7);
                assert_relative_eq!(expected.im, actual.im, max_relative = 1e-7);
            }
            assert_relative_eq!(zpk.k, k, max_relative = 1e-7);
        }
    }

    #[cfg(all(feature = "alloc", feature = "std"))]
    #[test]
    fn matches_scipy_iirfilter_butter_zpk() {
        let expected_zpk: ZpkFormatFilter<f64> = ZpkFormatFilter::new(
            vec![
                Complex::new(1., 0.),
                Complex::new(1., 0.),
                Complex::new(1., 0.),
                Complex::new(1., 0.),
                Complex::new(-1., 0.),
                Complex::new(-1., 0.),
                Complex::new(-1., 0.),
                Complex::new(-1., 0.),
            ],
            vec![
                Complex::new(0.98924866, -0.03710237),
                Complex::new(0.96189799, -0.03364097),
                Complex::new(0.96189799, 0.03364097),
                Complex::new(0.98924866, 0.03710237),
                Complex::new(0.93873849, 0.16792939),
                Complex::new(0.89956011, 0.08396115),
                Complex::new(0.89956011, -0.08396115),
                Complex::new(0.93873849, -0.16792939),
            ],
            2.6775767382597835e-5,
        );
        let filter = iirfilter_dyn::<f64>(
            4,
            vec![10., 50.],
            None,
            None,
            Some(FilterBandType::Bandpass),
            Some(FilterType::Butterworth),
            Some(false),
            Some(FilterOutputType::Zpk),
            Some(1666.),
        );

        match filter {
            DigitalFilter::Zpk(zpk) => {
                assert_eq!(zpk.z.len(), expected_zpk.z.len());
                for (a, e) in zpk.z.iter().zip(expected_zpk.z.iter()) {
                    assert_relative_eq!(a.re, e.re, max_relative = 1e-6);
                    assert_relative_eq!(a.im, e.im, max_relative = 1e-6);
                }

                assert_eq!(zpk.p.len(), expected_zpk.p.len());
                for (a, e) in zpk.p.iter().zip(expected_zpk.p.iter()) {
                    assert_relative_eq!(a.re, e.re, max_relative = 1e-6);
                    assert_relative_eq!(a.im, e.im, max_relative = 1e-6);
                }

                assert_relative_eq!(zpk.k, expected_zpk.k, max_relative = 1e-8);
            }
            _ => panic!(),
        }
    }

    #[cfg(all(feature = "alloc", feature = "std"))]
    #[test]
    fn matches_scipy_iirfilter_butter_sos() {
        let filter = iirfilter_dyn::<f64>(
            4,
            vec![10., 50.],
            None,
            None,
            Some(FilterBandType::Bandpass),
            Some(FilterType::Butterworth),
            Some(false),
            Some(FilterOutputType::Sos),
            Some(1666.),
        );

        match filter {
            DigitalFilter::Sos(sos) => {
                // println!("{:?}", sos);

                let expected_sos = [
                    Sos::new(
                        [2.67757674e-05, 5.35515348e-05, 2.67757674e-05],
                        [1.00000000e+00, -1.79912022e+00, 8.16257861e-01],
                    ),
                    Sos::new(
                        [1.00000000e+00, 2.00000000e+00, 1.00000000e+00],
                        [1.00000000e+00, -1.87747699e+00, 9.09430241e-01],
                    ),
                    Sos::new(
                        [1.00000000e+00, -2.00000000e+00, 1.00000000e+00],
                        [1.00000000e+00, -1.92379599e+00, 9.26379467e-01],
                    ),
                    Sos::new(
                        [1.00000000e+00, -2.00000000e+00, 1.00000000e+00],
                        [1.00000000e+00, -1.97849731e+00, 9.79989489e-01],
                    ),
                ];

                assert_eq!(expected_sos.len(), sos.sos.len());
                for i in 0..sos.sos.len() {
                    let actual = sos.sos[i];
                    let expected = expected_sos[i];
                    assert_relative_eq!(actual.b[0], expected.b[0], max_relative = 1e-7);
                    assert_relative_eq!(actual.b[1], expected.b[1], max_relative = 1e-7);
                    assert_relative_eq!(actual.b[2], expected.b[2], max_relative = 1e-7);
                    assert_relative_eq!(actual.a[0], expected.a[0], max_relative = 1e-7);
                    assert_relative_eq!(actual.a[1], expected.a[1], max_relative = 1e-7);
                    assert_relative_eq!(actual.a[2], expected.a[2], max_relative = 1e-7);
                }
            }
            _ => panic!(),
        }
    }

    #[cfg(all(feature = "alloc", feature = "std"))]
    #[test]
    fn matches_scipy_iirfilter_butter_ba() {
        let filter = iirfilter_dyn::<f64>(
            4,
            vec![10., 50.],
            None,
            None,
            Some(FilterBandType::Bandpass),
            Some(FilterType::Butterworth),
            Some(false),
            Some(FilterOutputType::Ba),
            Some(1666.),
        );

        match filter {
            DigitalFilter::Ba(ba) => {
                let expected_b = [
                    2.67757674e-05,
                    0.00000000e+00,
                    -1.07103070e-04,
                    0.00000000e+00,
                    1.60654604e-04,
                    0.00000000e+00,
                    -1.07103070e-04,
                    0.00000000e+00,
                    2.67757674e-05,
                ];
                let expected_a = [
                    1.,
                    -7.57889051,
                    25.1632497,
                    -47.80506049,
                    56.83958432,
                    -43.31144279,
                    20.65538731,
                    -5.63674562,
                    0.67391808,
                ];

                assert_eq!(expected_b.len(), ba.b.len());
                assert_eq!(expected_a.len(), ba.a.len());
                assert_relative_eq!(ba.b[0], expected_b[0], max_relative = 1e-7);
                assert_relative_eq!(ba.b[1], expected_b[1], max_relative = 1e-7);
                assert_relative_eq!(ba.b[2], expected_b[2], max_relative = 1e-7);
                assert_relative_eq!(ba.b[3], expected_b[3], max_relative = 1e-7);
                assert_relative_eq!(ba.b[4], expected_b[4], max_relative = 1e-7);

                assert_relative_eq!(ba.a[0], expected_a[0], max_relative = 1e-7);
                assert_relative_eq!(ba.a[1], expected_a[1], max_relative = 1e-7);
                assert_relative_eq!(ba.a[2], expected_a[2], max_relative = 1e-7);
                assert_relative_eq!(ba.a[3], expected_a[3], max_relative = 1e-7);
                assert_relative_eq!(ba.a[4], expected_a[4], max_relative = 1e-7);
            }
            _ => panic!(),
        }
    }

    #[cfg(all(feature = "alloc", feature = "std"))]
    #[test]
    fn matches_scipy_iirfilter_butter_zpk_highpass() {
        //zo = [1. 1. 1. 1.]
        //po = [0.86788666-0.23258286j 0.76382075-0.08478723j 0.76382075+0.08478723j 0.86788666+0.23258286j]
        //ko = 0.6905166297398233
        let expected_zpk: ZpkFormatFilter<f64> = ZpkFormatFilter::new(
            vec![
                Complex::new(1., 0.),
                Complex::new(1., 0.),
                Complex::new(1., 0.),
                Complex::new(1., 0.),
            ],
            vec![
                Complex::new(0.86788666, -0.23258286),
                Complex::new(0.76382075, -0.08478723),
                Complex::new(0.76382075, 0.08478723),
                Complex::new(0.86788666, 0.23258286),
            ],
            0.6905166297398233,
        );
        let filter = iirfilter_dyn::<f64>(
            4,
            vec![90.],
            None,
            None,
            Some(FilterBandType::Highpass),
            Some(FilterType::Butterworth),
            Some(false),
            Some(FilterOutputType::Zpk),
            Some(2003.),
        );

        match filter {
            DigitalFilter::Zpk(zpk) => {
                assert_eq!(zpk.z.len(), expected_zpk.z.len());
                for (a, e) in zpk.z.iter().zip(expected_zpk.z.iter()) {
                    assert_relative_eq!(a.re, e.re, max_relative = 1e-6);
                    assert_relative_eq!(a.im, e.im, max_relative = 1e-6);
                }

                assert_eq!(zpk.p.len(), expected_zpk.p.len());
                for (a, e) in zpk.p.iter().zip(expected_zpk.p.iter()) {
                    assert_relative_eq!(a.re, e.re, max_relative = 1e-6);
                    assert_relative_eq!(a.im, e.im, max_relative = 1e-6);
                }

                assert_relative_eq!(zpk.k, expected_zpk.k, max_relative = 1e-8);
            }
            _ => panic!(),
        }
    }

    #[cfg(all(feature = "alloc", feature = "std"))]
    #[test]
    fn matches_scipy_iirfilter_butter_zpk_lowpass() {
        //z1 = [-1. -1. -1. -1.]
        //p1 = [0.86788666+0.23258286j 0.76382075+0.08478723j 0.76382075-0.08478723j 0.86788666-0.23258286j]
        //k1 = 0.0002815867605254161
        let expected_zpk: ZpkFormatFilter<f64> = ZpkFormatFilter::new(
            vec![
                Complex::new(-1., 0.),
                Complex::new(-1., 0.),
                Complex::new(-1., 0.),
                Complex::new(-1., 0.),
            ],
            vec![
                Complex::new(0.86788666, 0.23258286),
                Complex::new(0.76382075, 0.0847872),
                Complex::new(0.76382075, -0.08478723),
                Complex::new(0.86788666, -0.23258286),
            ],
            0.0002815867605254161,
        );
        let filter = iirfilter_dyn::<f64>(
            4,
            vec![90.],
            None,
            None,
            Some(FilterBandType::Lowpass),
            Some(FilterType::Butterworth),
            Some(false),
            Some(FilterOutputType::Zpk),
            Some(2003.),
        );

        match filter {
            DigitalFilter::Zpk(zpk) => {
                assert_eq!(zpk.z.len(), expected_zpk.z.len());
                for (a, e) in zpk.z.iter().zip(expected_zpk.z.iter()) {
                    assert_relative_eq!(a.re, e.re, max_relative = 1e-6);
                    assert_relative_eq!(a.im, e.im, max_relative = 1e-6);
                }

                assert_eq!(zpk.p.len(), expected_zpk.p.len());
                for (a, e) in zpk.p.iter().zip(expected_zpk.p.iter()) {
                    assert_relative_eq!(a.re, e.re, max_relative = 1e-6);
                    assert_relative_eq!(a.im, e.im, max_relative = 1e-6);
                }

                assert_relative_eq!(zpk.k, expected_zpk.k, max_relative = 1e-8);
            }
            _ => panic!(),
        }
    }
}
