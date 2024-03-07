use crate::fp::{Fp, FpConfig};

pub struct FrConfig;
impl FpConfig for FrConfig {
    const MODULUS: [u32; 8] = [
        0xf0000001u32,
        0x43e1f593u32,
        0x79b97091u32,
        0x2833e848u32,
        0x8181585du32,
        0xb85045b6u32,
        0xe131a029u32,
        0x30644e72u32,
    ];
    const GENERATOR: [u32; 8] = [5, 0, 0, 0, 0, 0, 0, 0];
    const OVERFLOW_ADJUSTMENT: [u32; 8] = [
        0x4ffffffbu32,
        0xac96341cu32,
        0x9f60cd29u32,
        0x36fc7695u32,
        0x7879462eu32,
        0x666ea36fu32,
        0x9a07df2fu32,
        0x0e0a77c1u32,
    ];
    const TWO_ADIC_ROOT_OF_UNITY: [u32; 8] = [
        0x725b19f0u32,
        0x9bd61b6eu32,
        0x41112ed4u32,
        0x402d111eu32,
        0x8ef62abcu32,
        0x00e0a7ebu32,
        0xa58a7e85u32,
        0x2a3c09f0u32,
    ];
    const SMALL_SUBGROUP_BASE: Option<u32> = Some(3);
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = Some(2);
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<[u32; 8]> = Some([
        0x83364d6eu32,
        0x93a86345u32,
        0x0d19f8d1u32,
        0x538ba277u32,
        0x62c26ea9u32,
        0xd2e4fc92u32,
        0x21940bdau32,
        0x11b08a06u32,
    ]);
}

pub type Fr = Fp<FrConfig>;

#[macro_export]
macro_rules! BN254_FR {
    ($c0:expr) => {{
        let (is_positive, limbs) = ark_ff_macros::to_sign_and_limbs!($c0);
        Fr::from_sign_and_limbs(is_positive, &limbs)
    }};
}

pub use BN254_FR;
