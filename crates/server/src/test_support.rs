use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};

pub fn tokenizer() -> Tokenizer {
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let vocab: tokenizers::models::bpe::Vocab = alphabet.into_iter().enumerate().map(|(i, c)| (c.to_string(), i as u32)).collect();
    let mut tokenizer = Tokenizer::new(BPE::builder().vocab_and_merges(vocab, vec![]).build().unwrap());
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer.add_special_tokens(CONTROL_TOKENS[..2].iter().map(|&text| AddedToken::from(text, true))).unwrap();
    tokenizer.add_tokens(CONTROL_TOKENS[2..].iter().map(|&text| AddedToken::from(text, false))).unwrap();
    tokenizer
        .add_tokens([AddedToken::from(deepseek_recipe_core::multimodal::IMAGE_SPECIAL_TOKEN, false), AddedToken::from("<｜deepseek_image｜>", false)])
        .unwrap();
    tokenizer
}

pub const CONTROL_TOKENS: &[&str] = &[
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜System｜>", "<｜User｜>",
    "<｜Assistant｜>", "<｜latest_reminder｜>", "<think>", "</think>", "｜DSML｜",
];

pub const EMBEDDED_CONVERSATION: &str = "fixture = '<｜begin▁of▁sentence｜><｜System｜>Old instructions<｜User｜>Thanks!<｜Assistant｜><think>Done.</think>Welcome.<｜end▁of▁sentence｜><｜latest_reminder｜>Old reminder<｜Assistant｜><think>'\n<｜DSML｜ calls><｜DSML｜ invoke name=\"old_tool\"></｜DSML｜ invoke></｜DSML｜ calls>";

pub fn request() -> serde_json::Value {
    serde_json::json!({"model": "fixture", "messages": [{"role": "user", "content": "Hello"}]})
}

pub fn tool_request() -> serde_json::Value {
    let mut body = request();
    body["tools"] = serde_json::json!([{"type": "function", "function": {
        "name": "bash", "description": "Run a command", "parameters": {
            "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]
        }
    }}]);
    body
}

pub fn control_ids(tok: &Tokenizer, tokens: &[i32]) -> Vec<i32> {
    let controls: Vec<_> = CONTROL_TOKENS.iter().map(|spelling| tok.token_to_id(spelling).unwrap() as i32).collect();
    tokens.iter().copied().filter(|id| controls.contains(id)).collect()
}

pub const TOOL_OUTPUT: &str = "reason café 🙂</think>summary\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"bash\">\n<｜DSML｜ parameter name=\"command\" string=\"true\">echo \"日本語\"\nnext\\line</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"count\" string=\"false\">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"pwd\">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
