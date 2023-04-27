use super::{chunk, UnLearnedKnowledge, UnlearnedFile};
use anyhow::Result;

pub fn parse_normal(file: UnlearnedFile) -> Result<UnLearnedKnowledge> {
    let content = std::fs::read_to_string(&file.path)?;
    let unlearned_chunk_vec = chunk(content.lines().map(|x| x.to_string()));
    let unlearned_knowledge = UnLearnedKnowledge {
        file_name: file.file_name,
        uploader: file.uploader,
        chunks: unlearned_chunk_vec,
    };
    Ok(unlearned_knowledge)
}
