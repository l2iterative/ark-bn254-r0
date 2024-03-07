use crate::fields::Fq;
use ark_ff::{AdditiveGroup, Fp2, Fp2Config};

pub struct Fq2Config;

impl Fp2Config for Fq2Config {
    type Fp = Fq;
    const NONRESIDUE: Self::Fp = Fq::MINUS_ONE;
    const FROBENIUS_COEFF_FP2_C1: &'static [Self::Fp] = &[Fq::ONE, Fq::MINUS_ONE];

    fn mul_fp_by_nonresidue_in_place(fe: &mut Self::Fp) -> &mut Self::Fp {
        fe.neg_in_place()
    }
}

pub type Fq2 = Fp2<Fq2Config>;
