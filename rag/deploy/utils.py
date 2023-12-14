# import libraries
import streamlit as st
from pathlib import Path
from haystack import Pipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.components.converters import MarkdownToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceTGIGenerator


def setup_indexing_pipe():
    """Pre-processes documents and loads them into document store. Returns document store and indexing pipeline."""

    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    document_dir = './data'
    document_paths = [file_path.absolute() for file_path in Path(document_dir).glob('*.md')]

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("converter", MarkdownToDocument())
    indexing_pipeline.add_component("cleaner", DocumentCleaner())
    indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=500, split_overlap=50))
    indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model_name_or_path="sentence-transformers/msmarco-distilbert-cos-v5"))
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
    indexing_pipeline.connect("converter", "cleaner")
    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    indexing_pipeline.run({
        "converter": {
            "sources": document_paths
        }
    })

    return document_store


def streaming_callback_function(streaming_chunk):
    """Function to return streaming chunks part by part."""

    for c in streaming_chunk:
        yield c


def get_prompt_template():
    """Returns prompt template for use in query pipeline."""

    template = """<s>[INST] You are a virtual assistant experienced in answering questions based on provided documents. Given the following context, answer the question at the end in a truthful and elaborate manner using only the information from the context and nothing else.

    Context: 
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{ question }}

    Answer: [/INST]"""

    return template


def get_hf_client(model_name):
    """Sets up Hugging Face Text Generation Inference client for use in query pipeline."""

    model = {'Mistral-7B': "mistralai/Mistral-7B-Instruct-v0.1", 'Llama-2-7B': "mistralai/Mistral-7B-Instruct-v0.1"}

    hf_client = HuggingFaceTGIGenerator(
        model=model[model_name],
        token=st.secrets["hf_token"],
        generation_kwargs={
            "max_new_tokens": 1024,
            "temperature": 0.01,
        },
        stop_words=["</s>"],
        streaming_callback=streaming_callback_function
    )

    return hf_client


def setup_query_pipe(model_name):
    """Sets up query pipeline to handle user questions."""

    document_store = setup_indexing_pipe()
    template = get_prompt_template()
    hf_client = get_hf_client(model_name)

    query_pipeline = Pipeline()

    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model_name_or_path="sentence-transformers/msmarco-distilbert-cos-v5"))
    query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
    query_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    query_pipeline.add_component("llm", hf_client)
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder", "llm")

    return query_pipeline
