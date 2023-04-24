use super::{chunk, UnLearnedKnowledge};
use anyhow::Result;
use pdfium_render::prelude::Pdfium;
use std::{fmt::Display, path::PathBuf};

#[derive(PartialEq, Clone)]
pub struct Pdf {
    pub file_name: String,
    pub uploader: String,
    pub path: PathBuf,
}

impl Display for Pdf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Pdf {{ file_name: {}, uploader: {}, path: {} }}",
            self.file_name,
            self.uploader,
            self.path.display()
        )
    }
}

impl Pdf {
    pub fn new(file_name: String, uploader: String, path: PathBuf) -> Self {
        Self {
            file_name,
            uploader,
            path,
        }
    }
}

impl From<Pdf> for Result<UnLearnedKnowledge> {
    fn from(pdf: Pdf) -> Self {
        let pdfium = Pdfium::new(Pdfium::bind_to_library("./libpdfium.so")?);
        let pdf_document = pdfium.load_pdf_from_file(&pdf.path, None)?;
        let mut pages = vec![];
        for page in pdf_document.pages().iter() {
            pages.push(page.text()?.all())
        }
        let unlearned_chunk_vec = chunk(pages.into_iter());
        let unlearned_knowledge = UnLearnedKnowledge {
            file_name: pdf.file_name,
            uploader: pdf.uploader,
            chunks: unlearned_chunk_vec,
        };
        Ok(unlearned_knowledge)
    }
}
