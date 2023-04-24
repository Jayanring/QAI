use super::storage::string_decode;
use anyhow::Result;
use opendal::Operator;
use std::cmp::Ordering;
use std::collections::HashMap;
use strsim::jaro_winkler;

#[derive(Copy, Clone, Debug)]
pub struct Matched {
    pub index: usize,
    pub len: usize,
    pub vector_index: usize,
    similarity: f32,
}

impl PartialOrd for Matched {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.similarity.partial_cmp(&other.similarity)
    }
}

impl Ord for Matched {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl PartialEq for Matched {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl Eq for Matched {}

pub fn match_top_n(map: &HashMap<usize, Vec<Vec<f32>>>, vector: &[f32]) -> Vec<Matched> {
    let mut top_n = Vec::new();
    let total_len: usize = map.values().map(|v| v.len()).sum();
    let n = total_len / 10;

    for (index, vec_list) in map.iter() {
        for (vector_index, vec) in vec_list.iter().enumerate() {
            let similarity = cosine_similarity(vector, vec);
            debug!(
                "index: {}, vector_index: {}, similarity: {}",
                index, vector_index, similarity
            );
            let matched = Matched {
                index: *index,
                vector_index,
                similarity,
                len: vec_list.len(),
            };

            if top_n.len() < n {
                top_n.push(matched);
                top_n.sort_unstable_by(|a, b| b.cmp(a));
            } else if let Some(last) = top_n.last() {
                if matched.similarity > last.similarity {
                    top_n.pop();
                    top_n.push(matched);
                    top_n.sort_unstable_by(|a, b| b.cmp(a));
                }
            }
        }
    }
    debug!("top_n: {:?}", top_n);

    top_n
}

pub async fn match_final(top_n: Vec<Matched>, query: &str, operator: Operator) -> Result<Matched> {
    let mut new_top_n = Vec::new();
    for mut matched in top_n {
        let content = string_decode(
            &operator
                .read(
                    &(matched.index.to_string()
                        + "/"
                        + &matched.vector_index.to_string()
                        + "/content"),
                )
                .await?,
        );
        let text_similarity = text_similarity(query, &content);
        matched.similarity += text_similarity;
        debug!(
            "index: {}, vector_index: {}, cosine_similarity: {}, text_similarity: {}, final_similarity: {}",
            matched.index,
            matched.vector_index,
            matched.similarity - text_similarity,
            text_similarity,
            matched.similarity
        );
        new_top_n.push(matched);
    }
    new_top_n.sort_unstable_by(|a, b| b.cmp(a));
    debug!("final matched: {:?}", new_top_n[0]);

    Ok(new_top_n[0])
}

fn text_similarity(s1: &str, s2: &str) -> f32 {
    jaro_winkler(s1, s2) as f32
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn magnitude(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let mag_a = magnitude(a);
    let mag_b = magnitude(b);

    dot / (mag_a * mag_b)
}
