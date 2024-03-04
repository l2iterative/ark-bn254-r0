use ark_std::{One, Zero};
use std::mem::MaybeUninit;
use std::ops::{Add, Mul};

pub(crate) const BIGINT_WIDTH_WORDS: usize = 8;
const OP_MULTIPLY: u32 = 0;

extern "C" {
    fn sys_bigint(
        result: *mut [u32; BIGINT_WIDTH_WORDS],
        op: u32,
        x: *const [u32; BIGINT_WIDTH_WORDS],
        y: *const [u32; BIGINT_WIDTH_WORDS],
        modulus: *const [u32; BIGINT_WIDTH_WORDS],
    );
}

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

const ONE: [u32; 8] = [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32];

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

#[derive(Clone, Copy)]
pub struct Fr {
    pub data: [u32; 8],
}

impl Fr {
    pub fn reduce(&self) -> [u32; 8] {
        let mut res = MaybeUninit::<[u32; 8]>::uninit();
        unsafe {
            sys_bigint(res.as_mut_ptr(), OP_MULTIPLY, &self.data, &ONE, &MODULUS);
        }
        let res = unsafe { res.assume_init() };
        for i in 0..8 {
            if res[7 - i] < MODULUS[7 - i] {
                return res;
            }
        }
        unreachable!()
    }
}

impl Default for Fr {
    fn default() -> Self {
        Fr::zero()
    }
}

impl PartialEq for Fr {
    fn eq(&self, other: &Self) -> bool {
        let left = self.reduce();
        let right = other.reduce();
        left == right
    }
}

impl Eq for Fr {}

impl Zero for Fr {
    #[inline]
    fn zero() -> Self {
        Self { data: [0u32; 8] }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.reduce() == [0u32; 8]
    }
}

impl One for Fr {
    fn one() -> Self {
        Self {
            data: [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32],
        }
    }

    fn is_one(&self) -> bool {
        self.reduce() == [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
    }
}

impl Add<Self> for Fr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut res = self.data.clone();

        let mut carry = add(&mut res, &rhs.data);
        while carry == 1 {
            carry = add(&mut res, &OVERFLOW_ADJUSTMENT);
        }

        unsafe {
            sys_bigint(&mut res, OP_MULTIPLY, &res, &ONE, &MODULUS);
        }

        return Self { data: res };
    }
}

impl Mul<Self> for Fr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = MaybeUninit::<[u32; 8]>::uninit();
        unsafe {
            sys_bigint(
                res.as_mut_ptr(),
                OP_MULTIPLY,
                &self.data,
                &rhs.data,
                &MODULUS,
            );
        }
        return Self {
            data: unsafe { res.assume_init() },
        };
    }
}

#[inline(always)]
pub fn add32_and_overflow(a: u32, b: u32, carry: u32) -> (u32, u32) {
    let v = (a as u64).wrapping_add(b as u64).wrapping_add(carry as u64);
    ((v >> 32) as u32, (v & 0xffffffff) as u32)
}

#[inline(always)]
pub fn carry32_and_overflow(a: u32, carry: u32) -> (u32, u32) {
    let (v, carry) = a.overflowing_add(carry);
    (carry as u32, v)
}

#[inline]
#[must_use]
pub fn add<const I: usize, const J: usize>(accm: &mut [u32; I], new: &[u32; J]) -> u32 {
    let mut carry = 0;
    (carry, accm[0]) = add32_and_overflow(accm[0], new[0], carry);
    for i in 1..J {
        (carry, accm[i]) = add32_and_overflow(accm[i], new[i], carry);
    }
    for i in J..I {
        (carry, accm[i]) = carry32_and_overflow(accm[i], carry);
    }
    carry
}
