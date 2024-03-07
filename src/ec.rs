use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{AdditiveGroup, Field, PrimeField};
use num_traits::Zero;

pub trait GLVConfigWithFastAffine: SWCurveConfig + GLVConfig {
    fn glv_mul_fast_affine(p: &Affine<Self>, k: Self::ScalarField) -> Projective<Self> {
        let ((sgn_k1, k1), (sgn_k2, k2)) = Self::scalar_decomposition(k);

        let mut b1 = *p;
        let mut b2 = Self::endomorphism(&p.into_group()).into_affine();

        if !sgn_k1 {
            b1 = -b1;
        }
        if !sgn_k2 {
            b2 = -b2;
        }

        let b1b2 = (b1 + &b2).into_affine();

        let b1 = (b1.x().unwrap(), b1.y().unwrap());
        let b2 = (b2.x().unwrap(), b2.y().unwrap());
        let b1b2 = (b1b2.x().unwrap(), b1b2.y().unwrap());

        let iter_k1 = ark_ff::BitIteratorBE::new(k1.into_bigint());
        let iter_k2 = ark_ff::BitIteratorBE::new(k2.into_bigint());

        let three_div_by_2 =
            Self::BaseField::from(3u32) * Self::BaseField::from(2u32).inverse().unwrap();

        let mut res: Option<(Self::BaseField, Self::BaseField)> = None;
        let mut skip_zeros = true;
        for pair in iter_k1.zip(iter_k2) {
            if skip_zeros && pair == (false, false) {
                skip_zeros = false;
                continue;
            }

            if res.is_some() {
                let (x, y) = res.as_ref().unwrap();

                let x_sqr: Self::BaseField = *x * x;
                let s: Self::BaseField = x_sqr * &three_div_by_2 * y.inverse().unwrap();
                let x2 = s.square() - x.double();
                let y2 = s * (*x - x2) - y;

                res = Some((x2, y2));
            }

            if !pair.0 && !pair.1 {
                continue;
            }

            let point_to_add = match pair {
                (true, false) => &b1,
                (false, true) => &b2,
                (true, true) => &b1b2,
                (false, false) => unreachable!(),
            };

            if res.is_some() {
                let (x1, y1) = res.unwrap();
                let x2 = &point_to_add.0;
                let y2 = &point_to_add.1;

                let s: Self::BaseField = (y1 - y2) * (x1 - x2).inverse().unwrap();
                let x3 = s.square() - x1 - x2;
                let y3 = s * (x1 - x3) - y1;

                res = Some((x3, y3));
            } else {
                res = Some(*point_to_add);
            }
        }

        if res.is_none() {
            Projective::zero()
        } else {
            let (x, y) = res.unwrap();
            Projective::new(x, y, Self::BaseField::ONE)
        }
    }
}
