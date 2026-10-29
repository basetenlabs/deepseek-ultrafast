use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Encoding {
    Raw,
    Fp4E4m3,
    Fp4E8m0,
}

impl Encoding {
    pub fn group(self) -> Option<usize> {
        match self {
            Self::Raw => None,
            Self::Fp4E4m3 => Some(16),
            Self::Fp4E8m0 => Some(32),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Segment {
    pub decoded_bytes: usize,
    pub encoding: Encoding,
}

impl Segment {
    pub fn raw(decoded_bytes: usize) -> Self {
        Self {
            decoded_bytes,
            encoding: Encoding::Raw,
        }
    }

    pub fn stored_bytes(self) -> Result<usize> {
        if let Some(group) = self.encoding.group() {
            ensure!(
                self.decoded_bytes.is_multiple_of(2 * group),
                "unaligned FP4 segment"
            );
            Ok(self.decoded_bytes / 4 + self.decoded_bytes / (2 * group))
        } else {
            Ok(self.decoded_bytes)
        }
    }

    pub fn validate_payload(self, bytes: &[u8]) -> Result<()> {
        ensure!(
            bytes.len() == self.stored_bytes()?,
            "invalid encoded segment size"
        );
        let Some(group) = self.encoding.group() else {
            return Ok(());
        };
        let data_bytes = self.decoded_bytes / 4;
        for (g, &scale) in bytes[data_bytes..].iter().enumerate() {
            match self.encoding {
                Encoding::Fp4E4m3 => ensure!((1..=126).contains(&scale), "invalid FP4 E4M3 scale"),
                Encoding::Fp4E8m0 => {
                    ensure!((1..=254).contains(&scale), "invalid FP4 E8M0 scale");
                    if scale >= 252 {
                        let s = f32::from_bits((scale as u32) << 23);
                        for &pair in &bytes[g * (group / 2)..(g + 1) * (group / 2)] {
                            for code in [pair & 7, (pair >> 4) & 7] {
                                let value =
                                    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][code as usize] * s;
                                ensure!(
                                    value <= f32::from_bits(0x7f7f0000),
                                    "FP4 value exceeds finite BF16"
                                );
                            }
                        }
                    }
                }
                Encoding::Raw => unreachable!(),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paper_formats_charge_data_and_scales_and_reject_bad_geometry() {
        for (decoded_bytes, encoding, stored) in
            [(1024, Encoding::Fp4E4m3, 288), (256, Encoding::Fp4E8m0, 68)]
        {
            let segment = Segment {
                decoded_bytes,
                encoding,
            };
            assert_eq!(segment.stored_bytes().unwrap(), stored);
            assert!(
                Segment {
                    decoded_bytes: decoded_bytes - 1,
                    encoding
                }
                .stored_bytes()
                .is_err()
            );
            assert!(segment.validate_payload(&vec![1; stored - 1]).is_err());
            segment.validate_payload(&vec![1; stored]).unwrap();
            assert_eq!(
                Segment {
                    decoded_bytes: 0,
                    encoding
                }
                .stored_bytes()
                .unwrap(),
                0
            );
        }
        assert_eq!(Segment::raw(usize::MAX).stored_bytes().unwrap(), usize::MAX);
    }

    #[test]
    fn scales_cannot_smuggle_zero_nan_negative_or_overflow_into_restore() {
        for encoding in [Encoding::Fp4E4m3, Encoding::Fp4E8m0] {
            let segment = Segment {
                decoded_bytes: 2 * encoding.group().unwrap(),
                encoding,
            };
            let mut data = vec![0; segment.stored_bytes().unwrap()];
            let end = data.len() - 1;
            for scale in [0, 127, 128, 255] {
                data[end] = scale;
                let valid = encoding == Encoding::Fp4E8m0 && matches!(scale, 127 | 128);
                assert_eq!(segment.validate_payload(&data).is_ok(), valid);
            }
            data[end] = 1;
            segment.validate_payload(&data).unwrap();
            if encoding == Encoding::Fp4E8m0 {
                data[end] = 254;
                data[0] = 0x77;
                assert!(segment.validate_payload(&data).is_err());
            }
        }
    }
}
