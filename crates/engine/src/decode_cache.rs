use anyhow::{Result, ensure};

pub(crate) fn boundary(
    prompt: usize,
    emitted: usize,
    committed: usize,
    max_seq: usize,
) -> Result<(usize, usize)> {
    ensure!(
        emitted > 0 && emitted - 1 <= committed,
        "emitted tokens exceed accepted decode output"
    );
    let end = prompt
        .checked_add(emitted)
        .ok_or_else(|| anyhow::anyhow!("decode cache length overflow"))?;
    ensure!(end <= max_seq, "decode cache exceeds sequence capacity");
    Ok((end, prompt + emitted.min(committed)))
}

pub(crate) fn tokens(
    prompt: &[i32],
    first: i32,
    emitted: usize,
    accepted: &[i32],
) -> Result<Vec<i32>> {
    ensure!(
        emitted > 0 && emitted - 1 <= accepted.len(),
        "missing accepted decode tokens"
    );
    let mut tokens = prompt.to_vec();
    tokens.push(first);
    tokens.extend_from_slice(&accepted[..emitted - 1]);
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitted_prefix_excludes_unemitted_lookahead_and_rejected_drafts() {
        assert_eq!(
            tokens(&[1, 2], 3, 3, &[4, 5, 90, 91]).unwrap(),
            [1, 2, 3, 4, 5]
        );
        assert_eq!(boundary(2, 3, 4, 100).unwrap(), (5, 5));
        assert_eq!(tokens(&[1, 2], 3, 1, &[90, 91]).unwrap(), [1, 2, 3]);
    }
}
