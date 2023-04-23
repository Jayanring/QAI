pub mod pdf;

use self::pdf::Pdf;
use crate::CHUNK_TOKENS;
use anyhow::Result;
use async_openai::types::EmbeddingInput;
use std::fmt::Display;
use tiktoken_rs::cl100k_base;

#[derive(PartialEq, Clone)]
pub enum FileType {
    Pdf(Pdf),
    NotSupported,
}

impl Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileType::Pdf(pdf) => write!(f, "{}", pdf),
            FileType::NotSupported => write!(f, "Not supported file type"),
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct UnLearnedChunk {
    pub content: String,
    pub page: usize,
}

#[derive(Clone)]
pub struct UnLearnedKnowledge {
    pub file_name: String,
    pub uploader: String,
    pub chunks: Vec<UnLearnedChunk>,
}

impl From<FileType> for Result<UnLearnedKnowledge> {
    fn from(file_type: FileType) -> Result<UnLearnedKnowledge> {
        match file_type {
            FileType::Pdf(pdf) => pdf.into(),
            FileType::NotSupported => unimplemented!("Not supported file type"),
        }
    }
}

impl From<UnLearnedKnowledge> for EmbeddingInput {
    fn from(val: UnLearnedKnowledge) -> Self {
        EmbeddingInput::StringArray(val.chunks.iter().map(|u| u.content.clone()).collect())
    }
}

fn chunk(iter: impl Iterator<Item = String>) -> Vec<UnLearnedChunk> {
    let bpe = cl100k_base().unwrap();
    let mut chunks = Vec::new();
    let mut legacy = UnLearnedChunk::default();
    let mut legacy_tokens = 0;
    for (page, content) in iter.enumerate() {
        let mut chunk = legacy;
        let mut current_tokens = legacy_tokens;
        let lines = content.lines();
        for line in lines {
            let new_line = line.to_string() + "\n";
            if chunk.content.is_empty() {
                chunk.page = page + 1;
            }
            current_tokens += bpe.encode_with_special_tokens(&new_line).len();
            chunk.content += &new_line;
            if current_tokens > *CHUNK_TOKENS {
                chunks.push(chunk);
                chunk = UnLearnedChunk::default();
                current_tokens = 0;
            }
        }
        legacy = chunk;
        legacy_tokens = current_tokens;
    }
    if !legacy.content.is_empty() {
        chunks.push(legacy);
    }

    chunks
}
