use deepseek_ultrafast_kv_transfer::TokenText;
use std::collections::HashMap;
use tokenizers::Tokenizer;

fn byte_of_char() -> HashMap<char, u8> {
    let mut bytes: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut chars = bytes.clone();
    let mut extra = 0;
    for b in 0..=255 {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(256 + extra);
            extra += 1;
        }
    }
    chars.into_iter().zip(bytes).map(|(c, b)| (char::from_u32(c).unwrap(), b as u8)).collect()
}

pub fn token_text(tok: &Tokenizer) -> TokenText {
    let added = tok.get_added_tokens_decoder();
    let vocab = tok.get_vocab(true);
    let n = vocab.values().max().map_or(0, |&id| id as usize + 1);
    let byte = byte_of_char();
    let mut tokens = vec![(Vec::new(), true); n];
    for (piece, id) in vocab {
        tokens[id as usize] = match added.get(&id) {
            Some(token) => (token.content.as_bytes().to_vec(), true),
            None => match piece.chars().map(|c| byte.get(&c).copied()).collect::<Option<Vec<u8>>>() {
                Some(bytes) => (bytes, false),
                None => (piece.into_bytes(), true),
            },
        };
    }
    TokenText::new(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::tokenizer;
    use deepseek_ultrafast_kv_transfer::{Prompt, align};

    fn text_bytes(text: &TokenText, id: i32) -> Vec<u8> {
        text.bytes(id).unwrap().0.to_vec()
    }

    #[test]
    fn token_bytes_spell_the_text() {
        let tok = tokenizer();
        let text = token_text(&tok);
        let s = "Tetris · pure vanilla ✓ 日本語\n\tend";
        let ids: Vec<i32> = tok.encode(s, false).unwrap().get_ids().iter().map(|&i| i as i32).collect();
        let spelled: Vec<u8> = ids.iter().flat_map(|&id| text_bytes(&text, id)).collect();
        assert_eq!(spelled, s.as_bytes());
        let mut prompt = ids.clone();
        prompt.push(ids[0]);
        assert!(align(&ids, Prompt::text(&prompt), &text).is_none(), "identical tokens: nothing to align");
    }

    #[test]
    fn added_tokens_are_atomic() {
        let tok = tokenizer();
        let text = token_text(&tok);
        let eos = tok.token_to_id("<｜end▁of▁sentence｜>").unwrap() as i32;
        let literal: Vec<i32> = "<｜end▁of▁sentence｜>".bytes()
            .map(|b| tok.token_to_id(&byte_of_char().iter().find(|(_, v)| **v == b).unwrap().0.to_string()).unwrap() as i32)
            .collect();
        assert!(align(&[eos], Prompt::text(&literal), &text).is_none());
        assert!(align(&literal, Prompt::text(&[eos]), &text).is_none());
    }

    #[test]
    fn checkpoint_tokens_decode_to_their_bytes() {
        let Ok(dir) = std::env::var("DSUF_TEST_CKPT") else {
            eprintln!("DSUF_TEST_CKPT unset: skipped");
            return;
        };
        let tok = Tokenizer::from_file(std::path::Path::new(&dir).join("tokenizer.json")).unwrap();
        let text = token_text(&tok);
        let (mut checked, mut atomic) = (0, 0);
        for id in 0..tok.get_vocab_size(true) as i32 {
            let Some((bytes, is_atomic)) = text.bytes(id) else { continue };
            if is_atomic {
                atomic += 1;
                continue;
            }
            if let Ok(s) = std::str::from_utf8(bytes) {
                assert_eq!(tok.decode(&[id as u32], false).unwrap(), s, "token {id}");
                checked += 1;
            }
        }
        eprintln!("{checked} tokens checked, {atomic} atomic");
        assert!(checked > 100_000);
    }
}
