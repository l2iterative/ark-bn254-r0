use ark_ff::BigInt;
use ark_ff::{AdditiveGroup, Field, SqrtPrecomputation};
use ark_serialize::{
    buffer_byte_size, CanonicalDeserialize, CanonicalDeserializeWithFlags, CanonicalSerialize,
    CanonicalSerializeWithFlags, Compress, EmptyFlags, Flags, SerializationError, Valid, Validate,
};
use ark_serialize::{Read, Write};
use ark_std::mem::MaybeUninit;
use ark_std::ops::{Add, Mul};
use ark_std::rand::Rng;
use ark_std::{One, Zero};
use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::ops::{AddAssign, DivAssign, MulAssign, Neg, Sub, SubAssign};
use zeroize::Zeroize;

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

const TWO: [u32; 8] = [2u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32];

const MINUS_ONE: [u32; 8] = [
    0xf0000000u32,
    0x43e1f593u32,
    0x79b97091u32,
    0x2833e848u32,
    0x8181585du32,
    0xb85045b6u32,
    0xe131a029u32,
    0x30644e72u32,
];

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

#[derive(Clone, Copy, Hash)]
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

    #[inline]
    pub fn from_bigint(other: BigInt<4>) -> Option<Self> {
        Some(Self {
            data: [
                (other.0[0] & 0xffffffff) as u32,
                ((other.0[0] >> 32) & 0xffffffff) as u32,
                (other.0[1] & 0xffffffff) as u32,
                ((other.0[1] >> 32) & 0xffffffff) as u32,
                (other.0[2] & 0xffffffff) as u32,
                ((other.0[2] >> 32) & 0xffffffff) as u32,
                (other.0[3] & 0xffffffff) as u32,
                ((other.0[3] >> 32) & 0xffffffff) as u32,
            ],
        })
    }

    #[inline]
    pub fn into_bigint(self) -> BigInt<4> {
        BigInt::<4>::new([
            (self.data[0] as u64) | ((self.data[1] as u64) << 32),
            (self.data[2] as u64) | ((self.data[3] as u64) << 32),
            (self.data[4] as u64) | ((self.data[5] as u64) << 32),
            (self.data[6] as u64) | ((self.data[7] as u64) << 32),
        ])
    }

    #[inline]
    pub fn is_geq_modulus(&self) -> bool {
        for i in 0..8 {
            if self.data[7 - i] < MODULUS[7 - i] {
                return false;
            }
        }
        return true;
    }
}

impl ark_std::fmt::Debug for Fr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        ark_std::fmt::Debug::fmt(&self.into_bigint(), f)
    }
}

impl Default for Fr {
    fn default() -> Self {
        Fr::zero()
    }
}

impl PartialEq for Fr {
    #[inline]
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

impl Display for Fr {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> ark_std::fmt::Result {
        let string = self.into_bigint().to_string();
        write!(f, "{}", string.trim_start_matches('0'))
    }
}

impl Add<Self> for Fr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl<'a> Add<&'a Self> for Fr {
    type Output = Fr;

    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl<'a> Add<&'a mut Self> for Fr {
    type Output = Fr;

    fn add(self, rhs: &'a mut Self) -> Self::Output {
        self.add(&*rhs)
    }
}

impl AddAssign<Self> for Fr {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs)
    }
}

impl<'a> AddAssign<&'a Self> for Fr {
    #[inline]
    fn add_assign(&mut self, rhs: &'a Self) {
        let mut carry = add(&mut self.data, &rhs.data);
        while carry == 1 {
            carry = add(&mut self.data, &OVERFLOW_ADJUSTMENT);
        }
    }
}

impl<'a> AddAssign<&'a mut Self> for Fr {
    #[inline]
    fn add_assign(&mut self, rhs: &'a mut Self) {
        self.add_assign(&*rhs)
    }
}

impl ark_std::iter::Sum<Self> for Fr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), core::ops::Add::add)
    }
}

impl<'a> ark_std::iter::Sum<&'a Self> for Fr {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), core::ops::Add::add)
    }
}

impl Sub<Self> for Fr {
    type Output = Fr;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<'a> Sub<&'a Self> for Fr {
    type Output = Fr;

    fn sub(mut self, rhs: &'a Self) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

impl<'a> Sub<&'a mut Self> for Fr {
    type Output = Fr;

    fn sub(self, rhs: &'a mut Self) -> Self::Output {
        self.sub(&*rhs)
    }
}

impl SubAssign<Self> for Fr {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs)
    }
}

impl<'a> SubAssign<&'a Self> for Fr {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.add_assign(rhs.neg())
    }
}

impl<'a> SubAssign<&'a mut Self> for Fr {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a mut Self) {
        self.sub_assign(&*rhs)
    }
}

impl Mul<Self> for Fr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<'a> Mul<&'a Fr> for Fr {
    type Output = Fr;

    fn mul(mut self, rhs: &'a Fr) -> Self::Output {
        self.mul_assign(rhs);
        self
    }
}

impl<'a> Mul<&'a mut Fr> for Fr {
    type Output = Fr;

    fn mul(self, rhs: &'a mut Fr) -> Self::Output {
        self.mul(&*rhs)
    }
}

impl MulAssign<Fr> for Fr {
    fn mul_assign(&mut self, rhs: Fr) {
        self.mul_assign(&rhs)
    }
}

impl<'a> MulAssign<&'a Fr> for Fr {
    fn mul_assign(&mut self, rhs: &'a Fr) {
        unsafe {
            sys_bigint(&mut self.data, OP_MULTIPLY, &self.data, &rhs.data, &MODULUS);
        }
    }
}

impl<'a> MulAssign<&'a mut Fr> for Fr {
    fn mul_assign(&mut self, rhs: &'a mut Fr) {
        unsafe {
            sys_bigint(&mut self.data, OP_MULTIPLY, &self.data, &rhs.data, &MODULUS);
        }
    }
}

impl PartialOrd for Fr {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Fr {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.into_bigint().cmp(&other.into_bigint())
    }
}

impl CanonicalSerializeWithFlags for Fr {
    fn serialize_with_flags<W: Write, F: Flags>(
        &self,
        mut writer: W,
        flags: F,
    ) -> Result<(), SerializationError> {
        if F::BIT_SIZE > 8 {
            return Err(SerializationError::NotEnoughSpace);
        }

        let output_byte_size = buffer_byte_size(254 + F::BIT_SIZE);

        let mut bytes = [
            (self.data[0] & 0xff) as u8,
            ((self.data[0] >> 8) & 0xff) as u8,
            ((self.data[0] >> 16) & 0xff) as u8,
            ((self.data[0] >> 24) & 0xff) as u8,
            (self.data[1] & 0xff) as u8,
            ((self.data[1] >> 8) & 0xff) as u8,
            ((self.data[1] >> 16) & 0xff) as u8,
            ((self.data[1] >> 24) & 0xff) as u8,
            (self.data[2] & 0xff) as u8,
            ((self.data[2] >> 8) & 0xff) as u8,
            ((self.data[2] >> 16) & 0xff) as u8,
            ((self.data[2] >> 24) & 0xff) as u8,
            (self.data[3] & 0xff) as u8,
            ((self.data[3] >> 8) & 0xff) as u8,
            ((self.data[3] >> 16) & 0xff) as u8,
            ((self.data[3] >> 24) & 0xff) as u8,
            (self.data[4] & 0xff) as u8,
            ((self.data[4] >> 8) & 0xff) as u8,
            ((self.data[4] >> 16) & 0xff) as u8,
            ((self.data[4] >> 24) & 0xff) as u8,
            (self.data[5] & 0xff) as u8,
            ((self.data[5] >> 8) & 0xff) as u8,
            ((self.data[5] >> 16) & 0xff) as u8,
            ((self.data[5] >> 24) & 0xff) as u8,
            (self.data[6] & 0xff) as u8,
            ((self.data[6] >> 8) & 0xff) as u8,
            ((self.data[6] >> 16) & 0xff) as u8,
            ((self.data[6] >> 24) & 0xff) as u8,
            (self.data[7] & 0xff) as u8,
            ((self.data[7] >> 8) & 0xff) as u8,
            ((self.data[7] >> 16) & 0xff) as u8,
            ((self.data[7] >> 24) & 0xff) as u8,
            0u8,
        ];

        if output_byte_size == 32 {
            bytes[31] |= flags.u8_bitmask();
            writer.write_all(&bytes[0..32])?;
        } else {
            bytes[32] = flags.u8_bitmask();
            writer.write_all(&bytes)?;
        }

        Ok(())
    }

    fn serialized_size_with_flags<F: Flags>(&self) -> usize {
        buffer_byte_size(254 + F::BIT_SIZE)
    }
}

impl CanonicalSerialize for Fr {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        self.serialize_with_flags(writer, EmptyFlags)
    }

    #[inline]
    fn serialized_size(&self, _compress: Compress) -> usize {
        self.serialized_size_with_flags::<EmptyFlags>()
    }
}

impl CanonicalDeserializeWithFlags for Fr {
    fn deserialize_with_flags<R: Read, F: Flags>(
        mut reader: R,
    ) -> Result<(Self, F), SerializationError> {
        if F::BIT_SIZE > 8 {
            return Err(SerializationError::NotEnoughSpace);
        }

        let output_byte_size = buffer_byte_size(254 + F::BIT_SIZE);

        let mut all_bytes = MaybeUninit::<[u8; 33]>::uninit();

        if output_byte_size == 32 {
            unsafe {
                reader.read_exact(&mut *(all_bytes.as_mut_ptr())[0..32])?;
            }
        } else {
            unsafe {
                reader.read_exact(&mut *(all_bytes.as_mut_ptr()))?;
            }
        }

        let mut all_bytes = unsafe { all_bytes.assume_init() };

        let flags = if output_byte_size == 32 {
            F::from_u8_remove_flags(&mut all_bytes[31])
                .ok_or(SerializationError::UnexpectedFlags)?
        } else {
            F::from_u8_remove_flags(&mut all_bytes[32])
                .ok_or(SerializationError::UnexpectedFlags)?
        };

        let res = Self {
            data: [
                (all_bytes[0] as u32)
                    | ((all_bytes[1] as u32) << 8)
                    | ((all_bytes[2] as u32) << 16)
                    | ((all_bytes[3] as u32) << 24),
                (all_bytes[4] as u32)
                    | ((all_bytes[5] as u32) << 8)
                    | ((all_bytes[6] as u32) << 16)
                    | ((all_bytes[7] as u32) << 24),
                (all_bytes[8] as u32)
                    | ((all_bytes[9] as u32) << 8)
                    | ((all_bytes[10] as u32) << 16)
                    | ((all_bytes[11] as u32) << 24),
                (all_bytes[12] as u32)
                    | ((all_bytes[13] as u32) << 8)
                    | ((all_bytes[14] as u32) << 16)
                    | ((all_bytes[15] as u32) << 24),
                (all_bytes[16] as u32)
                    | ((all_bytes[17] as u32) << 8)
                    | ((all_bytes[18] as u32) << 16)
                    | ((all_bytes[19] as u32) << 24),
                (all_bytes[20] as u32)
                    | ((all_bytes[21] as u32) << 8)
                    | ((all_bytes[22] as u32) << 16)
                    | ((all_bytes[23] as u32) << 24),
                (all_bytes[24] as u32)
                    | ((all_bytes[25] as u32) << 8)
                    | ((all_bytes[26] as u32) << 16)
                    | ((all_bytes[27] as u32) << 24),
                (all_bytes[28] as u32)
                    | ((all_bytes[29] as u32) << 8)
                    | ((all_bytes[30] as u32) << 16)
                    | ((all_bytes[31] as u32) << 24),
            ],
        };

        Ok((res, flags))
    }
}

impl Valid for Fr {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for Fr {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        Self::deserialize_with_flags::<R, EmptyFlags>(reader).map(|(r, _)| r)
    }
}

impl From<u128> for Fr {
    fn from(other: u128) -> Self {
        Self {
            data: [
                (other & 0xffffffff) as u32,
                ((other >> 32) & 0xffffffff) as u32,
                ((other >> 64) & 0xffffffff) as u32,
                ((other >> 96) & 0xffffffff) as u32,
                0,
                0,
                0,
                0,
            ],
        }
    }
}

impl From<i128> for Fr {
    fn from(other: i128) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl From<bool> for Fr {
    fn from(value: bool) -> Self {
        if value {
            Self::one()
        } else {
            Self::zero()
        }
    }
}

impl From<u64> for Fr {
    fn from(other: u64) -> Self {
        Self {
            data: [
                (other & 0xffffffff) as u32,
                ((other >> 32) & 0xffffffff) as u32,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }
    }
}

impl From<i64> for Fr {
    fn from(other: i64) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl From<u32> for Fr {
    fn from(other: u32) -> Self {
        Self {
            data: [other, 0, 0, 0, 0, 0, 0, 0],
        }
    }
}

impl From<i32> for Fr {
    fn from(other: i32) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl From<u16> for Fr {
    fn from(other: u16) -> Self {
        Self {
            data: [other as u32, 0, 0, 0, 0, 0, 0, 0],
        }
    }
}

impl From<i16> for Fr {
    fn from(other: i16) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl From<u8> for Fr {
    fn from(other: u8) -> Self {
        Self {
            data: [other as u32, 0, 0, 0, 0, 0, 0, 0],
        }
    }
}

impl From<i8> for Fr {
    fn from(other: i8) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl ark_std::rand::distributions::Distribution<Fr> for ark_std::rand::distributions::Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Fr {
        let mut slice: [u32; 8] = [
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample::<u32, _>(ark_std::rand::distributions::Standard) & 0x3fffffff,
        ];

        unsafe {
            sys_bigint(&mut slice, OP_MULTIPLY, &slice, &ONE, &MODULUS);
        }

        Fr { data: slice }
    }
}

impl Neg for Fr {
    type Output = Self;

    #[inline]
    #[must_use]
    fn neg(self) -> Self::Output {
        let mut res = MaybeUninit::<[u32; 8]>::uninit();
        unsafe {
            sys_bigint(
                res.as_mut_ptr(),
                OP_MULTIPLY,
                &self.data,
                &MINUS_ONE,
                &MODULUS,
            );
        }
        return Self {
            data: unsafe { res.assume_init() },
        };
    }
}

impl AdditiveGroup for Fr {
    type Scalar = Fr;
    const ZERO: Self = Self::zero();

    fn double_in_place(&mut self) -> &mut Self {
        unsafe {
            sys_bigint(&mut self.data, OP_MULTIPLY, &self.data, &TWO, &MODULUS);
        }
        self
    }

    fn neg_in_place(&mut self) -> &mut Self {
        unsafe {
            sys_bigint(
                &mut self.data,
                OP_MULTIPLY,
                &self.data,
                &MINUS_ONE,
                &MODULUS,
            );
        }
        self
    }
}

impl<'a> DivAssign<&'a Self> for Fr {
    fn div_assign(&mut self, rhs: &'a Self) {
        todo!()
    }
}

impl Field for Fr {
    type BasePrimeField = Self;
    type BasePrimeFieldIter = core::iter::Once<Self::BasePrimeField>;

    const SQRT_PRECOMP: Option<SqrtPrecomputation<Self>> =
        Some(SqrtPrecomputation::TonelliShanks {
            two_adicity: 28,
            quadratic_nonresidue_to_trace: Fr {
                data: [
                    0x725b19f0u32,
                    0x9bd61b6eu32,
                    0x41112ed4u32,
                    0x402d111eu32,
                    0x8ef62abcu32,
                    0x00e0a7ebu32,
                    0xa58a7e85u32,
                    0x2a3c09f0u32,
                ],
            },
            trace_of_modulus_minus_one_div_two: &[
                0xcdcb848a1f0fac9f,
                0x0c0ac2e9419f4243,
                0x098d014dc2822db4,
                0x183227397,
            ],
        });

    const ONE: Self = Self::one();

    fn extension_degree() -> u64 {
        1
    }

    fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
        elem
    }

    fn to_base_prime_field_elements(&self) -> Self::BasePrimeFieldIter {
        core::iter::once(*self)
    }

    #[inline]
    fn from_base_prime_field_elems(
        elems: impl IntoIterator<Item = Self::BasePrimeField>,
    ) -> Option<Self> {
        let mut elems = elems.into_iter();
        let elem = elems.next()?;
        if elems.next().is_some() {
            return None;
        }
        Some(elem)
    }

    #[inline]
    fn characteristic() -> &'static [u64] {
        &[
            0x43e1f593f0000001u64,
            0x2833e84879b97091u64,
            0xb85045b68181585du64,
            0x30644e72e131a029u64,
        ]
    }

    #[inline]
    fn from_random_bytes_with_flags<F: Flags>(bytes: &[u8]) -> Option<(Self, F)> {
        if F::BIT_SIZE > 8 {
            return None;
        }

        let output_byte_size = buffer_byte_size(254 + F::BIT_SIZE);

        if output_byte_size == 32 {
            let mut all_bytes = MaybeUninit::<[u8; 32]>::uninit();
            all_bytes.as_mut_ptr().copy_from_slice(bytes);
            let mut all_bytes = unsafe { all_bytes.assume_init() };

            let flags = F::from_u8_remove_flags(&mut all_bytes[31])
                .ok_or(SerializationError::UnexpectedFlags)?;

            Self::deserialize_compressed(&all_bytes[0..32])
                .ok()
                .and_then(|f| Some((f, flags)))
        } else {
            let mut all_bytes = MaybeUninit::<[u8; 33]>::uninit();
            all_bytes.as_mut_ptr().copy_from_slice(bytes);
            let mut all_bytes = unsafe { all_bytes.assume_init() };

            let flags = F::from_u8_remove_flags(&mut all_bytes[32])
                .ok_or(SerializationError::UnexpectedFlags)?;

            Self::deserialize_compressed(&all_bytes[0..32])
                .ok()
                .and_then(|f| Some((f, flags)))
        }
    }
}

impl Zeroize for Fr {
    #[inline]
    fn zeroize(&mut self) {
        self.data.zeroize();
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
