pub const DEFAULT_TEMPERATURE: f32 = 1.0;
pub const DEFAULT_TOP_P: f32 = 1.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_one() {
        assert_eq!((DEFAULT_TEMPERATURE, DEFAULT_TOP_P), (1.0, 1.0));
    }
}
