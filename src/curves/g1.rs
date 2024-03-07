use crate::ec::GLVConfigWithFastAffine;
use crate::{Fq, Fr, BN254_FQ, BN254_FR};
use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_ec::{bn, CurveConfig, CurveGroup};
use ark_ff::{BigInt, PrimeField};
use num_bigint::{BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, Zero};
use std::ops::AddAssign;

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Config;

pub type G1Affine = Affine<Config>;

impl CurveConfig for Config {
    type BaseField = Fq;
    type ScalarField = Fr;

    /// COFACTOR = 1
    const COFACTOR: &'static [u64] = &[0x1];

    /// COFACTOR_INV = COFACTOR^{-1} mod r = 1
    const COFACTOR_INV: Fr = Fr::ONE;
}

impl SWCurveConfig for Config {
    /// COEFF_A = 0
    const COEFF_A: Fq = Fq::ZERO;

    /// COEFF_B = 3
    const COEFF_B: Fq = BN254_FQ!("3");

    /// AFFINE_GENERATOR_COEFFS = (G1_GENERATOR_X, G1_GENERATOR_Y)
    const GENERATOR: G1Affine = G1Affine::new_unchecked(G1_GENERATOR_X, G1_GENERATOR_Y);

    #[inline(always)]
    fn mul_by_a(_: Self::BaseField) -> Self::BaseField {
        Self::BaseField::zero()
    }

    #[inline]
    fn mul_projective(
        p: &bn::G1Projective<crate::Config>,
        scalar: &[u64],
    ) -> bn::G1Projective<crate::Config> {
        let s = Self::ScalarField::from_sign_and_limbs(true, scalar);
        GLVConfigWithFastAffine::glv_mul_fast_affine(&p.into_affine(), s)
    }

    #[inline]
    fn mul_affine(base: &Affine<Self>, scalar: &[u64]) -> Projective<Self> {
        let s = Self::ScalarField::from_sign_and_limbs(true, scalar);
        GLVConfigWithFastAffine::glv_mul_fast_affine(&base, s)
    }
}

impl GLVConfigWithFastAffine for Config {}

impl GLVConfig for Config {
    const ENDO_COEFFS: &'static [Self::BaseField] = &[BN254_FQ!(
        "21888242871839275220042445260109153167277707414472061641714758635765020556616"
    )];

    const LAMBDA: Self::ScalarField =
        BN254_FR!("21888242871839275217838484774961031246154997185409878258781734729429964517155");

    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4] = [
        (false, BigInt!("147946756881789319000765030803803410728")),
        (true, BigInt!("9931322734385697763")),
        (false, BigInt!("9931322734385697763")),
        (false, BigInt!("147946756881789319010696353538189108491")),
    ];

    fn endomorphism(p: &Projective<Self>) -> Projective<Self> {
        let mut res = (*p).clone();
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
    fn endomorphism_affine(p: &Affine<Self>) -> Affine<Self> {
        let mut res = (*p).clone();
        res.x *= Self::ENDO_COEFFS[0];
        res
    }

    fn scalar_decomposition(
        k: Self::ScalarField,
    ) -> ((bool, Self::ScalarField), (bool, Self::ScalarField)) {
        let scalar: num_bigint::BigInt = k.into_bigint().into();

        let coeff_bigints: [num_bigint::BigInt; 4] = Self::SCALAR_DECOMP_COEFFS.map(|x| {
            num_bigint::BigInt::from_biguint(
                x.0.then_some(Sign::Plus).unwrap_or(Sign::Minus),
                x.1.into(),
            )
        });

        let [n11, n12, n21, n22] = coeff_bigints;

        let r = num_bigint::BigInt::from(Self::ScalarField::MODULUS);

        // beta = vector([k,0]) * self.curve.N_inv
        // The inverse of N is 1/r * Matrix([[n22, -n12], [-n21, n11]]).
        // so β = (k*n22, -k*n12)/r

        let beta_1 = {
            let (mut div, rem) = (&scalar * &n22).div_rem(&r);
            if (&rem + &rem) > r {
                div.add_assign(num_bigint::BigInt::one());
            }
            div
        };
        let beta_2 = {
            let (mut div, rem) = (&scalar * &n12).div_rem(&r);
            if (&rem + &rem) > r {
                div.add_assign(num_bigint::BigInt::one());
            }
            div
        };

        // b = vector([int(beta[0]), int(beta[1])]) * self.curve.N
        // b = (β1N11 + β2N21, β1N12 + β2N22) with the signs!
        //   = (b11   + b12  , b21   + b22)   with the signs!

        // b1
        let b11 = &beta_1 * &n11;
        let b12 = &beta_2 * &n21;
        let b1 = b11 + b12;

        // b2
        let b21 = &beta_1 * &n12;
        let b22 = &beta_2 * &n22;
        let b2 = b21 + b22;

        let k1 = &scalar - b1;
        let k1_abs = BigUint::try_from(k1.abs()).unwrap();

        // k2
        let k2 = -b2;
        let k2_abs = BigUint::try_from(k2.abs()).unwrap();

        (
            (k1.sign() == Sign::Plus, Self::ScalarField::from(k1_abs)),
            (k2.sign() == Sign::Plus, Self::ScalarField::from(k2_abs)),
        )
    }
}

/// G1_GENERATOR_X = 1
pub const G1_GENERATOR_X: Fq = Fq::ONE;

/// G1_GENERATOR_Y = 2
pub const G1_GENERATOR_Y: Fq = BN254_FQ!("2");
