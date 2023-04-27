pub mod normal;
pub mod pdf;

use crate::CHUNK_TOKENS;
use anyhow::Result;
use async_openai::types::EmbeddingInput;
use std::{fmt::Display, path::PathBuf};
use tiktoken_rs::cl100k_base;

use self::{normal::parse_normal, pdf::parse_pdf};

#[derive(PartialEq, Clone)]
pub enum FileType {
    Pdf,
    Normal,
}

impl Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileType::Pdf => write!(f, "PDF"),
            FileType::Normal => write!(f, "NORMAL"),
        }
    }
}

#[derive(Clone)]
pub struct UnlearnedFile {
    pub file_name: String,
    pub uploader: String,
    pub path: PathBuf,
    pub file_type: FileType,
}

impl Display for UnlearnedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {{ file_name: {}, uploader: {}, path: {} }}",
            self.file_type,
            self.file_name,
            self.uploader,
            self.path.display()
        )
    }
}

pub fn match_file(file_name: String, uploader: String, path: PathBuf) -> UnlearnedFile {
    let file_type = if file_name.ends_with(".pdf") {
        FileType::Pdf
    } else {
        FileType::Normal
    };
    UnlearnedFile {
        file_name,
        uploader,
        path,
        file_type,
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

impl From<UnlearnedFile> for Result<UnLearnedKnowledge> {
    fn from(file: UnlearnedFile) -> Result<UnLearnedKnowledge> {
        match file.file_type {
            FileType::Pdf => parse_pdf(file),
            FileType::Normal => parse_normal(file),
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
