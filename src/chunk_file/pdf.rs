use super::{chunk, UnLearnedKnowledge, UnlearnedFile};
use anyhow::Result;
use pdfium_render::prelude::Pdfium;

pub fn parse_pdf(file: UnlearnedFile) -> Result<UnLearnedKnowledge> {
    let pdfium = Pdfium::new(Pdfium::bind_to_library("./libpdfium.so")?);
    let pdf_document = pdfium.load_pdf_from_file(&file.path, None)?;
    let mut pages = vec![];
    for page in pdf_document.pages().iter() {
        pages.push(page.text()?.all())
    }
    let unlearned_chunk_vec = chunk(pages.into_iter());
    let unlearned_knowledge = UnLearnedKnowledge {
        file_name: file.file_name,
        uploader: file.uploader,
        chunks: unlearned_chunk_vec,
    };
    Ok(unlearned_knowledge)
}
