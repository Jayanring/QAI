use super::{chunk, insert_newlines, UnLearnedKnowledge, UnlearnedFile};
use anyhow::Result;
use docx_rust::document::BodyContent;
use docx_rust::DocxFile;

pub fn parse_docx(file: UnlearnedFile) -> Result<UnLearnedKnowledge> {
    let docx =
        DocxFile::from_file(file.path).map_err(|_| anyhow::anyhow!("Failed to read docx file"))?;
    let docx = docx
        .parse()
        .map_err(|_| anyhow::anyhow!("Failed to parse docx file"))?;

    let mut paras = vec![];
    for body in docx.document.body.content {
        if let BodyContent::Paragraph(para) = body {
            let s = para.text();
            if !s.is_empty() {
                paras.push(s);
            }
        }
    }
    let paras_with_newlines = insert_newlines(paras);
    let unlearned_chunk_vec = chunk(paras_with_newlines.into_iter());

    let unlearned_knowledge = UnLearnedKnowledge {
        file_name: file.file_name,
        uploader: file.uploader,
        chunks: unlearned_chunk_vec,
    };
    Ok(unlearned_knowledge)
}
