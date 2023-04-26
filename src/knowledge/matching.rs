use super::storage::string_decode;
use anyhow::Result;
use opendal::Operator;
use std::cmp::Ordering;
use std::collections::HashMap;

const TEXT_SIMILARITY_MIN: f64 = 0.4;
const TEXT_SIMILARITY_FACTOR: f32 = 0.22287;

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
    let n = total_len / 6;

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
        matched.similarity += text_similarity * TEXT_SIMILARITY_FACTOR;
        debug!(
            "index: {}, vector_index: {}, cosine_similarity: {}, text_similarity: {} * {}, final_similarity: {}",
            matched.index,
            matched.vector_index,
            matched.similarity - text_similarity * TEXT_SIMILARITY_FACTOR,
            text_similarity,
            TEXT_SIMILARITY_FACTOR,
            matched.similarity
        );
        new_top_n.push(matched);
    }
    new_top_n.sort_unstable_by(|a, b| b.cmp(a));
    debug!("final matched: {:?}", new_top_n[0]);

    Ok(new_top_n[0])
}

fn jaro_similarity_without_search_range(a: &str, b: &str) -> f64 {
    let a_chars = a.chars();
    let b_chars = b.chars();
    let a_len = a_chars.clone().count();
    let b_len = b_chars.clone().count();

    if a_len == 0 && b_len == 0 {
        return 1.0;
    } else if a_len == 0 || b_len == 0 {
        return 0.0;
    } else if a_len == 1 && b_len == 1 {
        return if a.eq(b) { 1.0 } else { 0.0 };
    }

    let mut b_consumed = vec![false; b_len];

    let mut matches = 0.0;
    let mut transpositions = 0.0;
    let mut b_match_index = 0;

    for (_, a_elem) in a_chars.enumerate() {
        for (j, b_elem) in b_chars.clone().enumerate() {
            if a_elem == b_elem && !b_consumed[j] {
                b_consumed[j] = true;
                matches += 1.0;

                if j < b_match_index {
                    transpositions += 1.0;
                }
                b_match_index = j;

                break;
            }
        }
    }

    if matches == 0.0 {
        TEXT_SIMILARITY_MIN
    } else {
        let res = (1.0 / 3.0)
            * ((matches / a_len as f64)
                + (matches / b_len as f64)
                + ((matches - transpositions) / matches));
        res.max(TEXT_SIMILARITY_MIN)
    }
}

fn text_similarity(s1: &str, s2: &str) -> f32 {
    jaro_similarity_without_search_range(s1, s2) as f32
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
