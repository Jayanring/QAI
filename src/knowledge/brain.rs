use super::matching::{match_final, match_top_n};
use super::storage::Storage;
use crate::chunk_file::UnLearnedKnowledge;
use crate::{CHUNK_HEAD, CHUNK_TAIL};
use anyhow::Result;
use async_openai::types::{
    ChatCompletionRequestMessageArgs, CreateChatCompletionRequestArgs, Role,
};
use async_openai::{types::CreateEmbeddingRequestArgs, Client};
use futures::stream::SplitSink;
use futures::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, Semaphore};
use warp::ws::{Message, WebSocket};

#[derive(Clone)]
pub struct BrainMetadata {
    pub name: String,
    pub admin: String,
}

#[derive(Clone, Default)]
pub struct Knowledge {
    list: Vec<String>,
    vectors: HashMap<usize, Vec<Vec<f32>>>,
}

#[derive(Clone)]
pub(crate) struct Brain {
    pub _metadata: BrainMetadata,
    openai_client: Client,
    pub storage: Storage,
    pub knowledge: Arc<RwLock<Knowledge>>,
    semaphore: Arc<Semaphore>,
}

impl Brain {
    pub async fn new(name: String, admin: String) -> Self {
        let brain = Self {
            _metadata: BrainMetadata { name, admin },
            openai_client: Client::new(),
            storage: Storage::new().await,
            knowledge: Arc::new(RwLock::new(Knowledge::default())),
            semaphore: Arc::new(Semaphore::new(1)),
        };
        let vectors = brain.storage.get_vectors().await.unwrap();
        let list = brain.storage.get_list().await.unwrap();
        {
            let mut write = brain.knowledge.write().await;
            write.vectors = vectors;
            write.list = list.clone();
        }
        info!("brain init, recover: len: {}, list: {:?}", list.len(), list);
        brain
    }

    pub async fn index(
        &self,
        unlearned_knowledge: UnLearnedKnowledge,
    ) -> Result<(), Box<dyn Error>> {
        let file_name = unlearned_knowledge.file_name.clone();

        // get vectors
        let start = Instant::now();
        let request = CreateEmbeddingRequestArgs::default()
            .model("text-embedding-ada-002")
            .input(unlearned_knowledge.clone())
            .build()?;
        let vectors = self
            .openai_client
            .embeddings()
            .create(request)
            .await?
            .data
            .into_iter()
            .map(|e| e.embedding)
            .collect::<Vec<_>>();
        let elapsed = start.elapsed().as_secs_f64();
        info!("embedding {} spends {}s", file_name, elapsed);

        // persist and update
        let permit = self.semaphore.acquire().await;
        let start = Instant::now();
        let index = self
            .storage
            .store(unlearned_knowledge, vectors.clone())
            .await?;
        let elapsed = start.elapsed().as_secs_f64();
        info!("index: {} persist {} spends {}s", index, file_name, elapsed);
        {
            let mut write = self.knowledge.write().await;
            write.vectors.insert(index, vectors);
            write.list.push(file_name.clone());
        }
        drop(permit);

        Ok(())
    }

    pub async fn query(
        &self,
        query: String,
        tx: &mut SplitSink<WebSocket, Message>,
    ) -> Result<(), Box<dyn Error>> {
        // embedding query
        let start = Instant::now();
        let request = CreateEmbeddingRequestArgs::default()
            .model("text-embedding-ada-002")
            .input(query.clone())
            .build()?;
        info!("request embedding, wait for response...");
        let vector = self.openai_client.embeddings().create(request).await?.data[0]
            .embedding
            .clone();
        let elapsed = start.elapsed().as_secs_f64();
        info!("embedding query: {} spends {}s", query, elapsed);

        // match
        let start = Instant::now();
        let matched = self.retrieve(&vector, &query).await?;
        let _uploader = matched.uploader;
        let file_name = matched.file_name;
        let context = matched
            .chunks
            .iter()
            .map(|c| c.content.clone())
            .collect::<String>();
        let page = matched.chunks[0].page;
        let elapsed = start.elapsed().as_secs_f64();
        info!(
            "match query: {} spends {}s, matched file_name: {}, page: {}",
            query, elapsed, file_name, page
        );

        // query openai
        let start = Instant::now();
        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(1200u16)
            .model("gpt-3.5-turbo")
            .temperature(0.0)
            .messages([
                ChatCompletionRequestMessageArgs::default()
                    .role(Role::System)
                    .content("你是一名客服经理，从已知中提取相关信息，详细且礼貌地回答问题。")
                    .build()?,
                ChatCompletionRequestMessageArgs::default()
                    .role(Role::User)
                    .content("已知：".to_string() + &context + "\n" + "问题：" + &query)
                    .build()?,
            ])
            .build()?;
        info!("send query to openai, wait for response...");
        let mut stream = self.openai_client.chat().create_stream(request).await?;

        tx.send(Message::text(":\n")).await?;
        debug!("query: {} replying...", query);
        while let Some(res) = stream.next().await {
            if let Ok(response) = res {
                for c in response.choices.iter() {
                    if let Some(content) = &c.delta.content {
                        let _ = tx.send(Message::text(content)).await;
                    }
                }
            }
        }
        let addition_info = format!("（详见 {} ，第 {} 页）", file_name, page,);
        tx.send(Message::text("\n\n".to_string() + &addition_info))
            .await?;
        let elapsed = start.elapsed().as_secs_f64();
        info!("query openai: {} spends {}s", query, elapsed);

        Ok(())
    }

    async fn retrieve(&self, vector: &[f32], query: &str) -> Result<UnLearnedKnowledge> {
        let top_n = {
            let map = &self.knowledge.read().await.vectors;
            match_top_n(map, vector)
        };
        let matched = match_final(top_n, query, self.storage.operator.clone()).await?;

        let start_index = {
            let pre = (*CHUNK_HEAD).min(matched.vector_index);
            matched.vector_index - pre
        };
        let end_index = (matched.vector_index + *CHUNK_TAIL).min(matched.len);
        let vector_indexs = (start_index..=end_index).collect::<Vec<_>>();

        // index 0 records matched page
        let matched_relative_index = matched.vector_index - start_index;
        let mut unlearned_knowledge = self.storage.load(matched.index, vector_indexs).await?;
        let matched_page = unlearned_knowledge.chunks[matched_relative_index].page;
        unlearned_knowledge.chunks[0].page = matched_page;
        Ok(unlearned_knowledge)
    }

    pub async fn get_list(&self) -> Vec<String> {
        let read = self.knowledge.read().await;
        read.list.clone()
    }
}
