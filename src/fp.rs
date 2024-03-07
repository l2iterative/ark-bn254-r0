use crate::{sys_bigint, sys_untrusted_mod_inv, sys_untrusted_mod_sqrt};
use ark_ff::{
    AdditiveGroup, BigInt, BigInteger, FftField, Field, LegendreSymbol, PrimeField,
    SqrtPrecomputation,
};
use ark_serialize::{
    buffer_byte_size, CanonicalDeserialize, CanonicalDeserializeWithFlags, CanonicalSerialize,
    CanonicalSerializeWithFlags, Compress, EmptyFlags, Flags, SerializationError, Valid, Validate,
};
use ark_serialize::{Read, Write};
use ark_std::cmp::Ordering;
use ark_std::fmt::{Display, Formatter};
use ark_std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use ark_std::rand::Rng;
use ark_std::str::FromStr;
use derivative::Derivative;
use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use zeroize::Zeroize;

pub(crate) const BIGINT_WIDTH_WORDS: usize = 8;
const OP_MULTIPLY: u32 = 0;

const ONE: [u32; 8] = [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32];

const TWO: [u32; 8] = [2u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32];

pub trait FpConfig: Send + Sync + 'static + Sized {
    const MODULUS: [u32; 8];
    const GENERATOR: [u32; 8];
    const OVERFLOW_ADJUSTMENT: [u32; 8];
    const TWO_ADIC_ROOT_OF_UNITY: [u32; 8];
    const SMALL_SUBGROUP_BASE: Option<u32>;
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32>;
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<[u32; 8]>;
    const MODULUS_MINUS_ONE: [u32; 8] = [
        Self::MODULUS[0] - 1,
        Self::MODULUS[1],
        Self::MODULUS[2],
        Self::MODULUS[3],
        Self::MODULUS[4],
        Self::MODULUS[5],
        Self::MODULUS[6],
        Self::MODULUS[7],
    ];
}

#[derive(Derivative)]
#[derivative(Hash(bound = ""), Clone(bound = ""), Copy(bound = ""))]
pub struct Fp<P: FpConfig> {
    pub data: [u32; 8],
    pub phantom: PhantomData<P>,
}

impl<P: FpConfig> Fp<P> {
    pub const ONE: Self = Self {
        data: ONE,
        phantom: PhantomData,
    };

    pub const ZERO: Self = Self {
        data: [0u32; 8],
        phantom: PhantomData,
    };

    pub const MINUS_ONE: Self = Self {
        data: P::MODULUS_MINUS_ONE,
        phantom: PhantomData,
    };

    pub fn reduce(&self) -> [u32; 8] {
        let mut res = MaybeUninit::<[u32; 8]>::uninit();
        unsafe {
            sys_bigint(res.as_mut_ptr(), OP_MULTIPLY, &self.data, &ONE, &P::MODULUS);
        }
        let res = unsafe { res.assume_init() };
        for i in 0..8 {
            if res[7 - i] < P::MODULUS[7 - i] {
                return res;
            }
        }

        unreachable!()
    }

    #[inline]
    pub fn is_geq_modulus(&self) -> bool {
        for i in 0..8 {
            if self.data[7 - i] < P::MODULUS[7 - i] {
                return false;
            }
        }
        return true;
    }

    #[inline]
    pub const fn from_sign_and_limbs(is_positive: bool, limbs: &[u64]) -> Self {
        let len = limbs.len();
        let mut buffer = [0u32; 8];

        if len >= 1 {
            buffer[0] = (limbs[0] & 0xffffffff) as u32;
            buffer[1] = ((limbs[0] >> 32) & 0xffffffff) as u32;
        }

        if len >= 2 {
            buffer[2] = (limbs[1] & 0xffffffff) as u32;
            buffer[3] = ((limbs[1] >> 32) & 0xffffffff) as u32;
        }

        if len >= 3 {
            buffer[4] = (limbs[2] & 0xffffffff) as u32;
            buffer[5] = ((limbs[2] >> 32) & 0xffffffff) as u32;
        }

        if len == 4 {
            buffer[6] = (limbs[3] & 0xffffffff) as u32;
            buffer[7] = ((limbs[3] >> 32) & 0xffffffff) as u32;
        }

        let res = Self {
            data: buffer,
            phantom: PhantomData,
        };
        if is_positive {
            res
        } else {
            res.const_neg()
        }
    }

    #[inline]
    pub const fn const_neg(self) -> Self {
        let (res, _) = sub_and_borrow(&P::MODULUS, &self.data);
        Self {
            data: res,
            phantom: PhantomData,
        }
    }
}

#[macro_export]
macro_rules! const_for {
    (($i:ident in $start:tt..$end:tt)  $code:expr ) => {{
        let mut $i = $start;
        while $i < $end {
            $code
            $i += 1;
        }
    }};
}

impl<P: FpConfig> ark_std::fmt::Debug for Fp<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        ark_std::fmt::Debug::fmt(&self.into_bigint(), f)
    }
}

impl<P: FpConfig> Default for Fp<P> {
    fn default() -> Self {
        Fp::zero()
    }
}

impl<P: FpConfig> PartialEq for Fp<P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let left = self.reduce();
        let right = other.reduce();
        left == right
    }
}

impl<P: FpConfig> Eq for Fp<P> {}

impl<P: FpConfig> Zero for Fp<P> {
    #[inline]
    fn zero() -> Self {
        Self {
            data: [0u32; 8],
            phantom: PhantomData,
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.reduce() == [0u32; 8]
    }
}

impl<P: FpConfig> One for Fp<P> {
    fn one() -> Self {
        Self {
            data: [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32],
            phantom: PhantomData,
        }
    }

    fn is_one(&self) -> bool {
        self.reduce() == [1u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
    }
}

impl<P: FpConfig> Display for Fp<P> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> ark_std::fmt::Result {
        let string = self.into_bigint().to_string();
        write!(f, "{}", string.trim_start_matches('0'))
    }
}

impl<P: FpConfig> Add<Self> for Fp<P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl<'a, P: FpConfig> Add<&'a Self> for Fp<P> {
    type Output = Self;

    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl<'a, P: FpConfig> Add<&'a mut Self> for Fp<P> {
    type Output = Self;

    fn add(self, rhs: &'a mut Self) -> Self::Output {
        self.add(&*rhs)
    }
}

impl<P: FpConfig> AddAssign<Self> for Fp<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs)
    }
}

impl<'a, P: FpConfig> AddAssign<&'a Self> for Fp<P> {
    #[inline]
    fn add_assign(&mut self, rhs: &'a Self) {
        let mut carry = add(&mut self.data, &rhs.data);
        while carry == 1 {
            carry = add(&mut self.data, &P::OVERFLOW_ADJUSTMENT);
        }
        let mut res = MaybeUninit::<[u32; 8]>::uninit();
        unsafe {
            sys_bigint(res.as_mut_ptr(), OP_MULTIPLY, &self.data, &ONE, &P::MODULUS);
        }
        self.data = unsafe { res.assume_init() }
    }
}

impl<'a, P: FpConfig> AddAssign<&'a mut Self> for Fp<P> {
    #[inline]
    fn add_assign(&mut self, rhs: &'a mut Self) {
        self.add_assign(&*rhs)
    }
}

impl<P: FpConfig> ark_std::iter::Sum<Self> for Fp<P> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), core::ops::Add::add)
    }
}

impl<'a, P: FpConfig> ark_std::iter::Sum<&'a Self> for Fp<P> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), core::ops::Add::add)
    }
}

impl<P: FpConfig> Sub<Self> for Fp<P> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<'a, P: FpConfig> Sub<&'a Self> for Fp<P> {
    type Output = Self;

    fn sub(mut self, rhs: &'a Self) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

impl<'a, P: FpConfig> Sub<&'a mut Self> for Fp<P> {
    type Output = Self;

    fn sub(self, rhs: &'a mut Self) -> Self::Output {
        self.sub(&*rhs)
    }
}

impl<P: FpConfig> SubAssign<Self> for Fp<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs)
    }
}

impl<'a, P: FpConfig> SubAssign<&'a Self> for Fp<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.add_assign(rhs.neg())
    }
}

impl<'a, P: FpConfig> SubAssign<&'a mut Self> for Fp<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: &'a mut Self) {
        self.sub_assign(&*rhs)
    }
}

impl<P: FpConfig> Mul<Self> for Fp<P> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<'a, P: FpConfig> Mul<&'a Self> for Fp<P> {
    type Output = Self;

    fn mul(mut self, rhs: &'a Self) -> Self::Output {
        self.mul_assign(rhs);
        self
    }
}

impl<'a, P: FpConfig> Mul<&'a mut Self> for Fp<P> {
    type Output = Self;

    fn mul(self, rhs: &'a mut Self) -> Self::Output {
        self.mul(&*rhs)
    }
}

impl<P: FpConfig> MulAssign<Self> for Fp<P> {
    fn mul_assign(&mut self, rhs: Self) {
        self.mul_assign(&rhs)
    }
}

impl<'a, P: FpConfig> MulAssign<&'a Self> for Fp<P> {
    fn mul_assign(&mut self, rhs: &'a Self) {
        let mut res = MaybeUninit::<[u32; 8]>::uninit();
        unsafe {
            sys_bigint(
                res.as_mut_ptr(),
                OP_MULTIPLY,
                &self.data,
                &rhs.data,
                &P::MODULUS,
            );
        }
        self.data = unsafe { res.assume_init() }
    }
}

impl<'a, P: FpConfig> MulAssign<&'a mut Self> for Fp<P> {
    fn mul_assign(&mut self, rhs: &'a mut Self) {
        self.mul_assign(&*rhs)
    }
}

impl<P: FpConfig> PartialOrd for Fp<P> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: FpConfig> Ord for Fp<P> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.into_bigint().cmp(&other.into_bigint())
    }
}

impl<P: FpConfig> CanonicalSerializeWithFlags for Fp<P> {
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

impl<P: FpConfig> CanonicalSerialize for Fp<P> {
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

impl<P: FpConfig> CanonicalDeserializeWithFlags for Fp<P> {
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
                reader.read_exact(&mut (*all_bytes.as_mut_ptr())[0..32])?;
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
            phantom: PhantomData,
        };

        Ok((res, flags))
    }
}

impl<P: FpConfig> Valid for Fp<P> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<P: FpConfig> CanonicalDeserialize for Fp<P> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        Self::deserialize_with_flags::<R, EmptyFlags>(reader).map(|(r, _)| r)
    }
}

impl<P: FpConfig> From<u128> for Fp<P> {
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
            phantom: PhantomData,
        }
    }
}

impl<P: FpConfig> From<i128> for Fp<P> {
    fn from(other: i128) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl<P: FpConfig> From<bool> for Fp<P> {
    fn from(value: bool) -> Self {
        if value {
            Self::one()
        } else {
            Self::zero()
        }
    }
}

impl<P: FpConfig> From<u64> for Fp<P> {
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
            phantom: PhantomData,
        }
    }
}

impl<P: FpConfig> From<i64> for Fp<P> {
    fn from(other: i64) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl<P: FpConfig> From<u32> for Fp<P> {
    fn from(other: u32) -> Self {
        Self {
            data: [other, 0, 0, 0, 0, 0, 0, 0],
            phantom: PhantomData,
        }
    }
}

impl<P: FpConfig> From<i32> for Fp<P> {
    fn from(other: i32) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl<P: FpConfig> From<u16> for Fp<P> {
    fn from(other: u16) -> Self {
        Self {
            data: [other as u32, 0, 0, 0, 0, 0, 0, 0],
            phantom: PhantomData,
        }
    }
}

impl<P: FpConfig> From<i16> for Fp<P> {
    fn from(other: i16) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl<P: FpConfig> From<u8> for Fp<P> {
    fn from(other: u8) -> Self {
        Self {
            data: [other as u32, 0, 0, 0, 0, 0, 0, 0],
            phantom: PhantomData,
        }
    }
}

impl<P: FpConfig> From<i8> for Fp<P> {
    fn from(other: i8) -> Self {
        let abs = Self::from(other.unsigned_abs());
        if other.is_positive() {
            abs
        } else {
            -abs
        }
    }
}

impl<P: FpConfig> ark_std::rand::distributions::Distribution<Fp<P>>
    for ark_std::rand::distributions::Standard
{
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Fp<P> {
        let slice: [u32; 8] = [
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample(ark_std::rand::distributions::Standard),
            rng.sample::<u32, _>(ark_std::rand::distributions::Standard) & 0x3fffffff,
        ];

        let mut res = MaybeUninit::<[u32; 8]>::uninit();

        unsafe {
            sys_bigint(res.as_mut_ptr(), OP_MULTIPLY, &slice, &ONE, &P::MODULUS);
        }

        Fp {
            data: unsafe { res.assume_init() },
            phantom: PhantomData,
        }
    }
}

impl<P: FpConfig> Neg for Fp<P> {
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
                &P::MODULUS_MINUS_ONE,
                &P::MODULUS,
            );
        }
        return Self {
            data: unsafe { res.assume_init() },
            phantom: PhantomData,
        };
    }
}

impl<P: FpConfig> AdditiveGroup for Fp<P> {
    type Scalar = Self;
    const ZERO: Self = Fp {
        data: [0u32; 8],
        phantom: PhantomData,
    };

    fn double_in_place(&mut self) -> &mut Self {
        let mut res = MaybeUninit::<[u32; 8]>::uninit();
        unsafe {
            sys_bigint(res.as_mut_ptr(), OP_MULTIPLY, &self.data, &TWO, &P::MODULUS);
        }
        self.data = unsafe { res.assume_init() };
        self
    }

    fn neg_in_place(&mut self) -> &mut Self {
        let mut res = MaybeUninit::<[u32; 8]>::uninit();
        unsafe {
            sys_bigint(
                res.as_mut_ptr(),
                OP_MULTIPLY,
                &self.data,
                &P::MODULUS_MINUS_ONE,
                &P::MODULUS,
            );
        }
        self.data = unsafe { res.assume_init() };
        self
    }
}

impl<P: FpConfig> Div<Self> for Fp<P> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl<'a, P: FpConfig> Div<&'a Self> for Fp<P> {
    type Output = Self;

    fn div(mut self, rhs: &'a Self) -> Self::Output {
        self.div_assign(rhs);
        self
    }
}

impl<'a, P: FpConfig> Div<&'a mut Self> for Fp<P> {
    type Output = Self;

    fn div(self, rhs: &'a mut Self) -> Self::Output {
        self.div(&*rhs)
    }
}

impl<P: FpConfig> DivAssign for Fp<P> {
    fn div_assign(&mut self, rhs: Self) {
        self.div_assign(&rhs)
    }
}

impl<'a, P: FpConfig> DivAssign<&'a Self> for Fp<P> {
    fn div_assign(&mut self, rhs: &'a Self) {
        self.mul_assign(&rhs.inverse().unwrap())
    }
}

impl<'a, P: FpConfig> DivAssign<&'a mut Self> for Fp<P> {
    fn div_assign(&mut self, rhs: &'a mut Self) {
        self.div_assign(&*rhs)
    }
}

impl<P: FpConfig> core::iter::Product<Self> for Fp<P> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), Mul::mul)
    }
}

impl<'a, P: FpConfig> core::iter::Product<&'a Self> for Fp<P> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), Mul::mul)
    }
}

impl<P: FpConfig> From<Fp<P>> for BigUint {
    fn from(value: Fp<P>) -> Self {
        BigUint::from_bytes_le(&value.into_bigint().to_bytes_le())
    }
}

impl<P: FpConfig> Field for Fp<P> {
    type BasePrimeField = Self;
    type BasePrimeFieldIter = core::iter::Once<Self::BasePrimeField>;

    const SQRT_PRECOMP: Option<SqrtPrecomputation<Self>> = None;

    const ONE: Self = Fp {
        data: ONE,
        phantom: PhantomData,
    };

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
            P::MODULUS[0] as u64 | ((P::MODULUS[1] as u64) << 32),
            P::MODULUS[2] as u64 | ((P::MODULUS[3] as u64) << 32),
            P::MODULUS[4] as u64 | ((P::MODULUS[5] as u64) << 32),
            P::MODULUS[6] as u64 | ((P::MODULUS[7] as u64) << 32),
        ]
    }

    #[inline]
    fn from_random_bytes_with_flags<F: Flags>(bytes: &[u8]) -> Option<(Self, F)> {
        if F::BIT_SIZE > 8 {
            return None;
        }

        let output_byte_size = buffer_byte_size(254 + F::BIT_SIZE);

        if output_byte_size == 32 {
            let mut all_bytes = [0u8; 32];
            all_bytes[0..bytes.len()].copy_from_slice(bytes);

            let flags = F::from_u8_remove_flags(&mut all_bytes[31]);
            if flags.is_none() {
                return None;
            }

            Self::deserialize_compressed(&all_bytes[0..32])
                .ok()
                .and_then(|f| Some((f, flags.unwrap())))
        } else {
            let mut all_bytes = [0u8; 33];
            all_bytes[0..bytes.len()].copy_from_slice(bytes);

            let flags = F::from_u8_remove_flags(&mut all_bytes[32]);
            if flags.is_none() {
                return None;
            }

            Self::deserialize_compressed(&all_bytes[0..32])
                .ok()
                .and_then(|f| Some((f, flags.unwrap())))
        }
    }

    #[inline]
    fn legendre(&self) -> LegendreSymbol {
        if self.is_zero() {
            return LegendreSymbol::Zero;
        }

        let mut res = MaybeUninit::<[u32; 8]>::uninit();

        let a0 = unsafe {
            sys_untrusted_mod_sqrt(
                res.as_mut_ptr() as *mut u32,
                &self.data,
                &P::MODULUS,
                &P::GENERATOR,
            )
        };

        let res = unsafe { res.assume_init() };

        return if a0 == 0 {
            let mut res_squared = MaybeUninit::<[u32; 8]>::uninit();
            unsafe {
                sys_bigint(
                    res_squared.as_mut_ptr(),
                    OP_MULTIPLY,
                    &res,
                    &res,
                    &P::MODULUS,
                );
            }
            let res_squared = unsafe { res_squared.assume_init() };
            assert_eq!(res_squared, self.reduce());

            LegendreSymbol::QuadraticResidue
        } else {
            let mut res_squared = MaybeUninit::<[u32; 8]>::uninit();
            unsafe {
                sys_bigint(
                    res_squared.as_mut_ptr(),
                    OP_MULTIPLY,
                    &res,
                    &res,
                    &P::MODULUS,
                );
            }
            let res_squared = unsafe { res_squared.assume_init() };

            let mut res_mul = MaybeUninit::<[u32; 8]>::uninit();
            unsafe {
                sys_bigint(
                    res_mul.as_mut_ptr(),
                    OP_MULTIPLY,
                    &self.data,
                    &P::GENERATOR,
                    &P::MODULUS,
                );
            }
            let res_mul = unsafe { res_mul.assume_init() };
            assert_eq!(res_squared, res_mul);

            LegendreSymbol::QuadraticNonResidue
        };
    }

    #[must_use]
    fn sqrt(&self) -> Option<Self> {
        if self.is_zero() {
            return Some(Self::ZERO);
        }

        let mut res = MaybeUninit::<[u32; 8]>::uninit();

        let a0 = unsafe {
            sys_untrusted_mod_sqrt(
                res.as_mut_ptr() as *mut u32,
                &self.data,
                &P::MODULUS,
                &P::GENERATOR,
            )
        };

        let res = unsafe { res.assume_init() };

        return if a0 == 0 {
            let mut res_squared = MaybeUninit::<[u32; 8]>::uninit();
            unsafe {
                sys_bigint(
                    res_squared.as_mut_ptr(),
                    OP_MULTIPLY,
                    &res,
                    &res,
                    &P::MODULUS,
                );
            }
            let res_squared = unsafe { res_squared.assume_init() };
            assert_eq!(res_squared, self.reduce());

            Some(Self {
                data: res,
                phantom: PhantomData,
            })
        } else {
            let mut res_squared = MaybeUninit::<[u32; 8]>::uninit();
            unsafe {
                sys_bigint(
                    res_squared.as_mut_ptr(),
                    OP_MULTIPLY,
                    &res,
                    &res,
                    &P::MODULUS,
                );
            }
            let res_squared = unsafe { res_squared.assume_init() };

            let mut res_mul = MaybeUninit::<[u32; 8]>::uninit();
            unsafe {
                sys_bigint(
                    res_mul.as_mut_ptr(),
                    OP_MULTIPLY,
                    &self.data,
                    &P::GENERATOR,
                    &P::MODULUS,
                );
            }
            let res_mul = unsafe { res_mul.assume_init() };
            assert_eq!(res_squared, res_mul);

            None
        };
    }

    #[inline]
    fn square(&self) -> Self {
        self.mul(self)
    }

    #[inline]
    fn square_in_place(&mut self) -> &mut Self {
        let res = self.square();
        *self = res;
        self
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let mut res = MaybeUninit::<[u32; 8]>::uninit();
        unsafe {
            sys_untrusted_mod_inv(res.as_mut_ptr() as *mut u32, &self.data, &P::MODULUS);
        }
        let res_inv = unsafe { res.assume_init() };

        let mut res = MaybeUninit::<[u32; 8]>::uninit();
        unsafe {
            sys_bigint(
                res.as_mut_ptr(),
                OP_MULTIPLY,
                &self.data,
                &res_inv,
                &P::MODULUS,
            );
        }
        let res_check = unsafe { res.assume_init() };

        for i in 1..8 {
            assert_eq!(res_check[i], 0);
        }
        assert_eq!(res_check[0], 1);

        Some(Self {
            data: res_inv,
            phantom: PhantomData,
        })
    }

    #[inline]
    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        let res = self.inverse();
        if res.is_none() {
            return None;
        } else {
            *self = res.unwrap();
            return Some(self);
        }
    }

    #[inline]
    fn frobenius_map_in_place(&mut self, _: usize) {}
}

impl<P: FpConfig> FromStr for Fp<P> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use num_bigint::BigInt;
        use num_traits::Signed;

        let modulus = BigInt::from(Self::MODULUS);
        let mut a = BigInt::from_str(s).map_err(|_| ())? % &modulus;
        if a.is_negative() {
            a += modulus
        }
        BigUint::try_from(a)
            .map_err(|_| ())
            .and_then(TryFrom::try_from)
            .ok()
            .and_then(Self::from_bigint)
            .ok_or(())
    }
}

impl<P: FpConfig> FftField for Fp<P> {
    const GENERATOR: Self = Fp {
        data: P::GENERATOR,
        phantom: PhantomData,
    };
    const TWO_ADICITY: u32 = Self::MODULUS.two_adic_valuation();
    const TWO_ADIC_ROOT_OF_UNITY: Self = Fp {
        data: P::TWO_ADIC_ROOT_OF_UNITY,
        phantom: PhantomData,
    };
    const SMALL_SUBGROUP_BASE: Option<u32> = P::SMALL_SUBGROUP_BASE;
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = P::SMALL_SUBGROUP_BASE_ADICITY;
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<Self> = to_fp(P::LARGE_SUBGROUP_ROOT_OF_UNITY);
}

pub const fn to_fp<P: FpConfig>(v: Option<[u32; 8]>) -> Option<Fp<P>> {
    if v.is_none() {
        None
    } else {
        match v {
            Some(v) => Some(Fp {
                data: v,
                phantom: PhantomData,
            }),
            _ => unreachable!(),
        }
    }
}

impl<P: FpConfig> PrimeField for Fp<P> {
    type BigInt = BigInt<4>;
    const MODULUS: Self::BigInt = BigInt::<4>::new([
        (P::MODULUS[0] as u64) | ((P::MODULUS[1] as u64) << 32),
        (P::MODULUS[2] as u64) | ((P::MODULUS[3] as u64) << 32),
        (P::MODULUS[4] as u64) | ((P::MODULUS[5] as u64) << 32),
        (P::MODULUS[6] as u64) | ((P::MODULUS[7] as u64) << 32),
    ]);
    const MODULUS_MINUS_ONE_DIV_TWO: Self::BigInt = Self::MODULUS.divide_by_2_round_down();
    const MODULUS_BIT_SIZE: u32 = Self::MODULUS.const_num_bits();
    const TRACE: Self::BigInt = Self::MODULUS.two_adic_coefficient();
    const TRACE_MINUS_ONE_DIV_TWO: Self::BigInt = Self::MODULUS
        .two_adic_coefficient()
        .divide_by_2_round_down();

    fn from_bigint(other: Self::BigInt) -> Option<Self> {
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
            phantom: PhantomData,
        })
    }

    fn into_bigint(self) -> Self::BigInt {
        let reduced = self.reduce();
        BigInt::<4>::new([
            (reduced[0] as u64) | ((reduced[1] as u64) << 32),
            (reduced[2] as u64) | ((reduced[3] as u64) << 32),
            (reduced[4] as u64) | ((reduced[5] as u64) << 32),
            (reduced[6] as u64) | ((reduced[7] as u64) << 32),
        ])
    }
}

impl<P: FpConfig> Into<BigInt<4>> for Fp<P> {
    fn into(self) -> BigInt<4> {
        self.into_bigint()
    }
}

impl<P: FpConfig> From<BigInt<4>> for Fp<P> {
    fn from(value: BigInt<4>) -> Self {
        Self::from_bigint(value).unwrap()
    }
}

impl<P: FpConfig> From<BigUint> for Fp<P> {
    fn from(value: BigUint) -> Self {
        Self::from_le_bytes_mod_order(&value.to_bytes_le())
    }
}

impl<P: FpConfig> Zeroize for Fp<P> {
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

#[inline]
pub const fn sub_with_borrow(a: u32, b: u32, carry: u32) -> (u32, u32) {
    let res = ((a as u64).wrapping_add(0x100000000))
        .wrapping_sub(b as u64)
        .wrapping_sub(carry as u64);
    (
        (res & 0xffffffff) as u32,
        1u32.wrapping_sub((res >> 32) as u32),
    )
}

#[inline]
pub const fn sub_and_borrow(accu: &[u32; 8], new: &[u32; 8]) -> ([u32; 8], u32) {
    let mut accu = [
        accu[0], accu[1], accu[2], accu[3], accu[4], accu[5], accu[6], accu[7],
    ];

    let (cur, borrow) = accu[0].overflowing_sub(new[0]);
    accu[0] = cur;

    let mut borrow = borrow as u32;
    (accu[1], borrow) = sub_with_borrow(accu[1], new[1], borrow);
    (accu[2], borrow) = sub_with_borrow(accu[2], new[2], borrow);
    (accu[3], borrow) = sub_with_borrow(accu[3], new[3], borrow);
    (accu[4], borrow) = sub_with_borrow(accu[4], new[4], borrow);
    (accu[5], borrow) = sub_with_borrow(accu[5], new[5], borrow);
    (accu[6], borrow) = sub_with_borrow(accu[6], new[6], borrow);
    (accu[7], borrow) = sub_with_borrow(accu[7], new[7], borrow);

    (accu, borrow)
}
