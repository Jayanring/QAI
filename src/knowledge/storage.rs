// after chunk_file and embedding we got following struct and a Vec<Vec<f32>> corresponds to chunks
//
// pub struct UnLearnedChunk {
//     pub content: String,
//     pub page: usize,
// }
//
// pub struct UnLearnedKnowledge {
//     pub file_name: String,
//     pub uploader: String,
//     pub chunks: Vec<UnLearnedChunk>,
// }
//
// storage protocol:
// count -> count of knowledges
// [i]/name -> i file_name
// [i]/uploader -> uploader: String
// [i]/count -> count of chunks
// [i]/[j]/vector -> j chunk vector
// [i]/[j]/content -> j chunk content
// [i]/[j]/page -> j chunk page
//
// writes should be mutually exclusive, but one write and some reads are allowed to be concurrent

use std::collections::HashMap;

use crate::chunk_file::{UnLearnedChunk, UnLearnedKnowledge};
use anyhow::Result;
use byteorder::{ByteOrder, LittleEndian};
use opendal::services::Sled;
use opendal::Operator;

#[derive(Clone)]
pub struct Storage {
    pub operator: Operator,
}

impl Storage {
    pub async fn new() -> Self {
        let mut builder = Sled::default();
        builder.datadir("./storage");

        Storage {
            operator: Operator::new(builder).unwrap().finish(),
        }
    }

    pub async fn store(
        &self,
        unlearned_knowledge: UnLearnedKnowledge,
        vectors: Vec<Vec<f32>>,
    ) -> Result<usize> {
        if unlearned_knowledge.chunks.len() != vectors.len() {
            return Err(anyhow::anyhow!("chunk index not match"));
        }

        let index = usize_decode(
            &self
                .operator
                .read("count")
                .await
                .unwrap_or(0usize.to_be_bytes().to_vec()),
        );
        self.operator
            .write(
                &(index.to_string() + "/name"),
                unlearned_knowledge.file_name,
            )
            .await?;
        self.operator
            .write(
                &(index.to_string() + "/uploader"),
                unlearned_knowledge.uploader,
            )
            .await?;
        self.operator
            .write(
                &(index.to_string() + "/count"),
                vectors.len().to_be_bytes().to_vec(),
            )
            .await?;
        for (j, chunk) in unlearned_knowledge.chunks.iter().enumerate() {
            self.operator
                .write(
                    &(index.to_string() + "/" + &j.to_string() + "/vector"),
                    float_to_bytes(&vectors[j]),
                )
                .await?;
            self.operator
                .write(
                    &(index.to_string() + "/" + &j.to_string() + "/content"),
                    chunk.content.clone(),
                )
                .await?;
            self.operator
                .write(
                    &(index.to_string() + "/" + &j.to_string() + "/page"),
                    chunk.page.to_be_bytes().to_vec(),
                )
                .await?;
            debug!(
                "store chunk: j: {}, content: {}, page: {}",
                j, chunk.content, chunk.page
            );
        }
        self.operator
            .write("count", (index + 1).to_be_bytes().to_vec())
            .await?;

        Ok(index)
    }

    pub async fn load(
        &self,
        index: usize,
        vector_indexs: Vec<usize>,
    ) -> Result<UnLearnedKnowledge> {
        let file_name = string_decode(&self.operator.read(&(index.to_string() + "/name")).await?);
        let uploader = string_decode(
            &self
                .operator
                .read(&(index.to_string() + "/uploader"))
                .await?,
        );

        let mut chunks = vec![];
        for j in vector_indexs.iter() {
            let content = string_decode(
                &self
                    .operator
                    .read(&(index.to_string() + "/" + &j.to_string() + "/content"))
                    .await?,
            );
            let page = usize_decode(
                &self
                    .operator
                    .read(&(index.to_string() + "/" + &j.to_string() + "/page"))
                    .await?,
            );
            debug!("load chunk: j: {}, content: {}", j, content);
            let chunk = UnLearnedChunk { content, page };
            chunks.push(chunk);
        }
        let unlearned_knowledge = UnLearnedKnowledge {
            file_name,
            uploader,
            chunks,
        };

        Ok(unlearned_knowledge)
    }

    pub async fn get_vectors(&self) -> Result<HashMap<usize, Vec<Vec<f32>>>> {
        let mut map = HashMap::new();
        let index = usize_decode(
            &self
                .operator
                .read("count")
                .await
                .unwrap_or(0usize.to_be_bytes().to_vec()),
        );
        for i in 0..index {
            let mut vectors = vec![];
            let chunk_index = usize_decode(&self.operator.read(&(i.to_string() + "/count")).await?);
            for j in 0..chunk_index {
                let vector = bytes_to_float(
                    &self
                        .operator
                        .read(&(i.to_string() + "/" + &j.to_string() + "/vector"))
                        .await?,
                );
                vectors.push(vector);
            }
            map.insert(i, vectors);
        }

        Ok(map)
    }

    pub async fn get_list(&self) -> Result<Vec<String>> {
        let mut list = vec![];
        let index = usize_decode(
            &self
                .operator
                .read("count")
                .await
                .unwrap_or(0usize.to_be_bytes().to_vec()),
        );
        for i in 0..index {
            let file_name = string_decode(&self.operator.read(&(i.to_string() + "/name")).await?);
            list.push(file_name);
        }
        Ok(list)
    }
}

fn usize_decode(data: &[u8]) -> usize {
    usize::from_be_bytes(data.try_into().unwrap())
}

pub fn string_decode(bytes: &[u8]) -> String {
    let s = String::from_utf8_lossy(bytes);
    s.to_string()
}

fn float_to_bytes(float_vec: &[f32]) -> Vec<u8> {
    let mut byte_vec = vec![0u8; float_vec.len() * 4];
    LittleEndian::write_f32_into(float_vec, &mut byte_vec);
    byte_vec
}

fn bytes_to_float(byte_vec: &[u8]) -> Vec<f32> {
    let mut float_vec = vec![0f32; byte_vec.len() / 4];
    LittleEndian::read_f32_into(byte_vec, &mut float_vec);
    float_vec
}
