#[cfg(feature = "curve")]
mod curves;

#[cfg(feature = "curve")]
pub use curves::*;

mod fields;
use crate::fp::BIGINT_WIDTH_WORDS;
pub use fields::*;

mod ec;
mod fp;

extern "C" {
    pub(crate) fn sys_bigint(
        result: *mut [u32; BIGINT_WIDTH_WORDS],
        op: u32,
        x: *const [u32; BIGINT_WIDTH_WORDS],
        y: *const [u32; BIGINT_WIDTH_WORDS],
        modulus: *const [u32; BIGINT_WIDTH_WORDS],
    );
}

extern "C" {
    pub(crate) fn sys_untrusted_mod_inv(
        result: *mut u32,
        x: *const [u32; BIGINT_WIDTH_WORDS],
        modulus: *const [u32; BIGINT_WIDTH_WORDS],
    );
}

extern "C" {
    pub(crate) fn sys_untrusted_mod_sqrt(
        result: *mut u32,
        x: *const [u32; BIGINT_WIDTH_WORDS],
        modulus: *const [u32; BIGINT_WIDTH_WORDS],
        quadratic_nonresidue: *const [u32; BIGINT_WIDTH_WORDS],
    ) -> u32;
}
