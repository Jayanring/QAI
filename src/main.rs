mod chunk_file;
mod knowledge;

use anyhow::Result;
use chunk_file::pdf::Pdf;
use chunk_file::FileType;
use dotenv::dotenv;
use env_logger::Builder;
use futures::StreamExt;
use futures_util::stream::TryStreamExt;
use knowledge::brain::Brain;
use lazy_static::lazy_static;
use log::LevelFilter;
use pdfium_render::prelude::Pdfium;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use tiktoken_rs::cl100k_base;
use tokio::io::AsyncWriteExt;
use tokio::{
    fs,
    sync::mpsc::{channel, Receiver, Sender},
};
use warp::{multipart::FormData, ws::WebSocket, Buf, Filter, Rejection, Reply};

#[macro_use]
extern crate log;

lazy_static! {
    static ref CHUNK_TOKENS: usize = std::env::var("CHUNK_TOKENS")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    static ref CHUNK_HEAD: usize = std::env::var("CHUNK_HEAD")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    static ref CHUNK_TAIL: usize = std::env::var("CHUNK_TAIL")
        .unwrap()
        .parse::<usize>()
        .unwrap();
}

#[derive(Deserialize, Serialize)]
struct QueryRequest {
    query: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // read .env
    dotenv().ok();

    // init logger
    let log_level = std::env::var("RUST_LOG").unwrap();
    if log_level == "debug" {
        Builder::new()
            .filter(None, LevelFilter::Info)
            .filter(Some("qa::knowledge"), LevelFilter::Debug)
            .init();
    } else {
        env_logger::init();
    }

    // check dependencies
    Pdfium::bind_to_library("./libpdfium.so")?;
    let upload_path = PathBuf::from("./files");
    if !upload_path.exists() {
        fs::create_dir("./files").await?;
    }
    cl100k_base()?;
    info!("dependencies check succeed");

    let (file_type_sender, file_type_reciever) = channel(1);

    let brain = Arc::new(Brain::new("test".to_string(), "wjj".to_string()).await);
    let brain_for_query = Arc::clone(&brain);
    let brain_for_index = Arc::clone(&brain);

    tokio::spawn(async move {
        indexer(file_type_reciever, brain_for_index).await;
    });

    // if a file is uploaded but not indexed, re-index it
    let uploaded = read_file_names(upload_path).await?;
    let indexed = brain.get_list().await;
    let unindexed = uploaded
        .into_iter()
        .filter(|x| !indexed.contains(x))
        .collect::<Vec<String>>();
    for file_name in unindexed {
        let file_path = PathBuf::from("./files").join(file_name.clone());
        let file_type = match_file_type(file_name.clone(), "".to_string(), file_path);
        let _ = file_type_sender.send(file_type).await;
        info!("send re-index to indexer: {}", file_name);
    }

    let index_route = warp::path::end().and(warp::get()).and_then(index);

    let file_upload_route = warp::path("upload")
        .and(warp::post())
        .and(warp::multipart::form())
        .and(warp::any().map(move || file_type_sender.clone()))
        .and_then(handle_upload);

    let query_route = warp::path("ws")
        .and(warp::ws())
        .and(warp::query::<QueryRequest>())
        .and(warp::any().map(move || Arc::clone(&brain_for_query)))
        .map(|ws: warp::ws::Ws, query: QueryRequest, brain| {
            ws.on_upgrade(move |socket| handle_query(query, brain, socket))
        });

    let get_list_route = warp::path("get_list")
        .and(warp::get())
        .and(warp::any().map(move || Arc::clone(&brain)))
        .and_then(handle_get_list);

    let routes = index_route
        .or(query_route)
        .or(file_upload_route)
        .or(get_list_route);

    warp::serve(routes).run(([127, 0, 0, 1], 8080)).await;

    Ok(())
}

async fn index() -> Result<impl Reply, Rejection> {
    let index_html = include_str!("../index.html");
    Ok(warp::reply::html(index_html))
}

async fn handle_get_list(brain: Arc<Brain>) -> Result<impl Reply, Rejection> {
    let list = brain.get_list().await;
    info!("get get_list request return: {:?}", list);
    let result = "\n".to_string() + &list.join("\n");
    Ok(warp::reply::html(result))
}

async fn handle_query(query_request: QueryRequest, brain: Arc<Brain>, ws: WebSocket) {
    info!("get query request: {:?}", query_request.query.clone());

    let (mut tx, _) = ws.split();

    if let Err(e) = brain.query(query_request.query, &mut tx).await {
        warn!("handle query request failed: {}", e);
    };
}

async fn handle_upload(
    form: FormData,
    file_type_sender: Sender<FileType>,
) -> Result<impl warp::Reply, Infallible> {
    let mut stream = form.into_stream();

    while let Ok(Some(part)) = stream.try_next().await {
        if let Some(file_name) = part.filename().map(|f| f.to_string()) {
            // check if exists
            let file_path = PathBuf::from("./files").join(file_name.clone());
            if file_path.exists() {
                info!("get upload request: {} already uploaded", file_name);
                return Ok(warp::reply::html("File already uploaded"));
            }
            // check if supported
            let file_type = match_file_type(file_name.clone(), "".to_string(), file_path.clone());
            if file_type == FileType::NotSupported {
                info!(
                    "get upload request: {} type not supported",
                    file_name.clone()
                );
                return Ok(warp::reply::html("File type not supported"));
            }
            // write file
            let mut file = fs::File::create(file_path.clone()).await.unwrap(); // should not panic
            let mut part_stream = part.stream();
            while let Ok(Some(chunk)) = part_stream.try_next().await {
                if let Err(e) = file.write_all(chunk.chunk()).await {
                    error!("write {} failed: {}", file_name, e);
                    return Ok(warp::reply::html("Write file failed"));
                }
            }
            info!("get upload request: {} uploaded", file_name);
            // send to indexer
            let _ = file_type_sender.send(file_type).await;
        }
    }
    Ok(warp::reply::html("File uploaded successfully. Indexing..."))
}

fn match_file_type(file_name: String, uploader: String, path: PathBuf) -> FileType {
    if file_name.ends_with(".pdf") {
        FileType::Pdf(Pdf::new(file_name, uploader, path))
    } else {
        FileType::NotSupported
    }
}

async fn indexer(mut file_type_reciever: Receiver<FileType>, brain: Arc<Brain>) {
    info!("indexer start");
    while let Some(file_type) = file_type_reciever.recv().await {
        info!("indexer recieve file: {}", file_type);
        match file_type.clone().into() {
            Ok(unlearned) => {
                let file_name = unlearned.file_name.clone();
                match brain.index(unlearned).await {
                    Ok(_) => {
                        info!("indexer index {} succeed", file_name);
                    }
                    Err(e) => {
                        warn!("indexer index {} failed: {}", file_name, e);
                    }
                }
            }
            Err(e) => {
                warn!("indexer parse file: {} failed: {}", file_type, e);
            }
        }
    }
}

async fn read_file_names(dir: PathBuf) -> Result<Vec<String>> {
    let mut file_names = Vec::new();
    let mut entries = fs::read_dir(dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        if entry.file_type().await?.is_file() {
            file_names.push(entry.file_name().to_string_lossy().to_string());
        }
    }
    Ok(file_names)
}
