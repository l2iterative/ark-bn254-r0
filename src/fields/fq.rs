use crate::fp::{Fp, FpConfig};

pub struct FqConfig;
impl FpConfig for FqConfig {
    const MODULUS: [u32; 8] = [
        0xd87cfd47u32,
        0x3c208c16u32,
        0x6871ca8du32,
        0x97816a91u32,
        0x8181585du32,
        0xb85045b6u32,
        0xe131a029u32,
        0x30644e72u32,
    ];
    const GENERATOR: [u32; 8] = [3, 0, 0, 0, 0, 0, 0, 0];
    const OVERFLOW_ADJUSTMENT: [u32; 8] = [
        0xc58f0d9du32,
        0xd35d438du32,
        0xf5c70b3du32,
        0x0a78eb28u32,
        0x7879462cu32,
        0x666ea36fu32,
        0x9a07df2fu32,
        0x0e0a77c1u32,
    ];
    const TWO_ADIC_ROOT_OF_UNITY: [u32; 8] = Self::MODULUS_MINUS_ONE;
    const SMALL_SUBGROUP_BASE: Option<u32> = None;
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = None;
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<[u32; 8]> = None;
}

pub type Fq = Fp<FqConfig>;

#[macro_export]
macro_rules! BN254_FQ {
    ($c0:expr) => {{
        let (is_positive, limbs) = ark_ff_macros::to_sign_and_limbs!($c0);
        Fq::from_sign_and_limbs(is_positive, &limbs)
    }};
}

pub use BN254_FQ;
