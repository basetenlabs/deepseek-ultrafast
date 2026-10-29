use deepseek_ultrafast_kv_transfer::ImageSpan;
use deepseek_ultrafast_model::vision::Image;
use tokio::sync::OwnedSemaphorePermit;

#[derive(Debug)]
pub struct Images {
    pub items: Vec<Image>,
    pub spans: Vec<ImageSpan>,
    pub(crate) _permit: std::sync::Arc<OwnedSemaphorePermit>,
}
impl Images {
    pub fn new(items: Vec<Image>, permit: OwnedSemaphorePermit) -> Self {
        let spans = items.iter().map(span).collect();
        Self { items, spans, _permit: std::sync::Arc::new(permit) }
    }
    pub fn shifted(&self, at: usize, shift: isize) -> Self {
        let items: Vec<Image> = self.items.iter().map(|im| {
            let mut im = im.clone();
            if im.start >= at { im.start = im.start.wrapping_add_signed(shift); }
            im
        }).collect();
        let spans = items.iter().map(span).collect();
        Self { items, spans, _permit: self._permit.clone() }
    }
}

pub fn span(im: &Image) -> ImageSpan {
    ImageSpan { start: im.start, end: im.end(), height: im.height, width: im.width, digest: im.digest }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn image(start: usize, digest: u8) -> Image {
        Image { start, height: 3, width: 3, digest: [digest; 32], patches: Arc::from(vec![0u16; 9 * 3 * 14 * 14]) }
    }

    #[test]
    fn spans_carry_identity_not_pixels() {
        let im = image(10, 7);
        let s = span(&im);
        assert_eq!((s.start, s.end, s.height, s.width, s.digest), (10, 10 + im.len(), 3, 3, [7; 32]));
        assert_eq!(std::mem::size_of::<ImageSpan>(), 4 * std::mem::size_of::<usize>() + 32);
        assert_ne!(span(&image(11, 7)), s);
        assert_ne!(span(&Image { width: 4, ..image(10, 7) }), s);
        assert_ne!(span(&image(10, 8)), s);
    }
}
