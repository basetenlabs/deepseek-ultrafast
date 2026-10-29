use crate::{ImageSpan, Prompt};

pub struct TokenText {
    bytes: Vec<Box<[u8]>>,
    atomic: Vec<bool>,
}
impl TokenText {
    pub fn new(tokens: Vec<(Vec<u8>, bool)>) -> Self {
        let (bytes, atomic) = tokens.into_iter().map(|(b, a)| (b.into_boxed_slice(), a)).unzip();
        Self { bytes, atomic }
    }
    pub fn bytes(&self, id: i32) -> Option<(&[u8], bool)> {
        let id = usize::try_from(id).ok()?;
        Some((self.bytes.get(id)?, self.atomic[id]))
    }
}

#[derive(Clone, Debug)]
pub struct Aligned {
    pub tokens: Vec<i32>,
    pub images: Vec<ImageSpan>,
    pub at: usize,
    pub shift: isize,
}
impl Aligned {
    pub fn prompt(&self) -> Prompt<'_> {
        Prompt::new(&self.tokens, &self.images)
    }
}

pub fn align(stored: &[i32], prompt: Prompt, text: &TokenText) -> Option<Aligned> {
    let tokens = prompt.tokens;
    let d = stored.iter().zip(tokens).take_while(|(a, b)| a == b).count();
    if d >= stored.len() || d >= tokens.len() {
        return None;
    }
    let limit = prompt.images.iter().filter(|s| s.end > d).map(|s| s.start).min().unwrap_or(tokens.len());
    if limit <= d {
        return None;
    }
    let (mut i, mut j, mut oi, mut oj) = (d, d, 0, 0);
    let mut best = None;
    loop {
        if oi == 0 && oj == 0 && i > d {
            best = Some((i, j));
        }
        if i == stored.len() || j == limit {
            break;
        }
        let (Some((a, a_atomic)), Some((b, b_atomic))) = (text.bytes(stored[i]), text.bytes(tokens[j])) else { break };
        if a_atomic || b_atomic {
            if oi != 0 || oj != 0 || stored[i] != tokens[j] {
                break;
            }
            i += 1;
            j += 1;
            continue;
        }
        let (ra, rb) = (&a[oi..], &b[oj..]);
        let n = ra.len().min(rb.len());
        if n == 0 || ra[..n] != rb[..n] {
            break;
        }
        oi += n;
        oj += n;
        if oi == a.len() {
            i += 1;
            oi = 0;
        }
        if oj == b.len() {
            j += 1;
            oj = 0;
        }
    }
    let (i, j) = best?;
    let shift = i as isize - j as isize;
    let mut aligned = stored[..i].to_vec();
    aligned.extend_from_slice(&tokens[j..]);
    let images = prompt
        .images
        .iter()
        .map(|s| {
            if s.start >= j {
                ImageSpan { start: s.start.wrapping_add_signed(shift), end: s.end.wrapping_add_signed(shift), ..*s }
            } else {
                *s
            }
        })
        .collect();
    Some(Aligned { tokens: aligned, images, at: j, shift })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text() -> TokenText {
        TokenText::new(vec![
            (b"a".to_vec(), false),
            (b"b".to_vec(), false),
            (b"ab".to_vec(), false),
            (b"c".to_vec(), false),
            (b"<sep>".to_vec(), true),
            (b"abc".to_vec(), false),
            (b"<im>".to_vec(), true),
        ])
    }
    fn span(start: usize, end: usize) -> ImageSpan {
        ImageSpan { start, end, height: 1, width: 1, digest: [7; 32] }
    }

    #[test]
    fn takes_the_stored_spelling_of_the_same_bytes() {
        let stored = [0, 1, 3, 4];
        let prompt = [5, 4, 0];
        let a = align(&stored, Prompt::text(&prompt), &text()).unwrap();
        assert_eq!(a.tokens, vec![0, 1, 3, 4, 0]);
        assert_eq!((a.at, a.shift), (2, 2));
        let a = align(&[0, 1, 3], Prompt::text(&[2, 3, 4]), &text()).unwrap();
        assert_eq!(a.tokens, vec![0, 1, 3, 4]);
    }

    #[test]
    fn stops_at_differing_bytes_and_never_matches_atomic_tokens_by_spelling() {
        let a = align(&[2, 3], Prompt::text(&[0, 1, 1]), &text()).unwrap();
        assert_eq!(a.tokens, vec![2, 1]);
        assert!(align(&[0, 1], Prompt::text(&[0, 1, 3]), &text()).is_none());
        assert!(align(&[3], Prompt::text(&[0]), &text()).is_none());
        let t = TokenText::new(vec![(b"<sep>".to_vec(), true), (b"<sep>".to_vec(), false)]);
        assert!(align(&[0], Prompt::text(&[1]), &t).is_none());
    }

    #[test]
    fn keeps_images_out_of_the_rewritten_range_and_moves_later_ones() {
        let stored = [0, 1, 3, 4, 6];
        let prompt = [5, 4, 6, 0];
        let images = [span(2, 3)];
        let a = align(&stored, Prompt::new(&prompt, &images), &text()).unwrap();
        assert_eq!(a.tokens, vec![0, 1, 3, 4, 6, 0]);
        assert_eq!(a.images, vec![span(4, 5)]);
        let images = [span(0, 1)];
        assert!(align(&[0, 1, 3, 4], Prompt::new(&[6, 4], &images), &text()).is_none());
    }
}
