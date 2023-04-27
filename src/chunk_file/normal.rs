use super::{chunk, insert_newlines, UnLearnedKnowledge, UnlearnedFile};
use anyhow::Result;

pub fn parse_normal(file: UnlearnedFile) -> Result<UnLearnedKnowledge> {
    let content = std::fs::read_to_string(&file.path)?;
    let content_replace_windows_newline = content.replace("\r\n", "\n");
    let paras = content_replace_windows_newline
        .split('\n')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_owned())
        .collect::<Vec<_>>();
    let paras_with_newlines = insert_newlines(paras);
    let unlearned_chunk_vec = chunk(paras_with_newlines.into_iter());

    let unlearned_knowledge = UnLearnedKnowledge {
        file_name: file.file_name,
        uploader: file.uploader,
        chunks: unlearned_chunk_vec,
    };
    Ok(unlearned_knowledge)
}
