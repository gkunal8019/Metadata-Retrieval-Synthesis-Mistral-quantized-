# Metadata-Retrieval-Synthesis-Mistral-quantized-

Here's the updated documentation formatted as a solution, incorporating the concept of "Metadata Extraction for Better Retrieval + Synthesis":

---

## Solution Overview

This solution classifies fraud descriptions into predefined categories using the `llama_index` library and HuggingFace models. It utilizes Metadata Extraction for Better Retrieval + Synthesis to enhance the accuracy of the classification process.

## Dependencies

Ensure the following Python packages are installed:

```sh
pip install llama-index-llms-openai
pip install llama-index-readers-web
pip install llama-index-llms-huggingface
pip install bitsandbytes
pip install accelerate
pip install transformers
pip install llama-index
pip install llama-index-embeddings-huggingface
```

## Code Explanation

### Importing Libraries

```python
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
import nest_asyncio
import gradio as gr
```

### Configuration

#### Quantization Configuration

The Mistral 7B model is highly quantized for efficient processing.

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

#### Embedding and LLM Settings

```python
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    device_map="cuda",
)
```

### Node Parsing and Metadata Extraction

The solution leverages Metadata Extraction for Better Retrieval + Synthesis to improve classification results.

```python
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor

node_parser = TokenTextSplitter(separator=" ", chunk_size=256, chunk_overlap=128)

extractors_1 = [
    QuestionsAnsweredExtractor(questions=3, llm=llm, metadata_mode=MetadataMode.EMBED),
]

extractors_2 = [
    SummaryExtractor(summaries=["prev", "self", "next"], llm=llm),
    QuestionsAnsweredExtractor(questions=3, llm=llm, metadata_mode=MetadataMode.EMBED),
]
```

### Data Loading

```python
reader = SimpleDirectoryReader(input_dir="/env/", required_exts=[".pdf"])
nest_asyncio.apply()
docs = reader.load_data()
orig_nodes = node_parser.get_nodes_from_documents(docs)
nodes = orig_nodes[21:28]
```

### Ingestion Pipeline

```python
from llama_index.core.ingestion import IngestionPipeline

pipeline = IngestionPipeline(transformations=[node_parser, *extractors_1])
nodes_1 = pipeline.run(nodes=nodes, in_place=False, show_progress=True)

pipeline = IngestionPipeline(transformations=[node_parser, *extractors_2])
nodes_2 = pipeline.run(nodes=nodes, in_place=False, show_progress=True)
```

### Creating Indexes and Query Engines

```python
index0 = VectorStoreIndex(orig_nodes)
index1 = VectorStoreIndex(orig_nodes[:20] + nodes_1 + orig_nodes[28:])
query_engine0 = index0.as_query_engine(similarity_top_k=1)
query_engine1 = index1.as_query_engine(similarity_top_k=1)
```

### Function to Identify Fraud Category

```python
def get_fraud_category(query, query_engine0, query_engine1):
    query_str = (
        f"""Identify the category for the description below. Use this data:
        Categories:
        - 'Aadhar Enabled Payment System (AEPS) fraud/ Biometric Cloning'
        - 'Online Loan Fraud'
        - 'Courier/Parcel Scam'
        - 'Online Shopping/E-commerce Frauds'
        - 'Malware Attack'
        - 'Tech Support Scam/Customer Care Scam'
        Respond with only the category.
        Description: {query}."""
    )

    response0 = query_engine0.query(query_str)
    response1 = query_engine1.query(query_str)
    
    result0 = display_response(response0, source_length=1000, show_source=True, show_source_metadata=True)
    result1 = display_response(response1, source_length=1000, show_source=True, show_source_metadata=True)
    
    return result0
```

### Gradio Interface

```python
iface = gr.Interface(fn=get_fraud_category, inputs="text", outputs="text", title="Fraud Category Identifier", description="Enter a description to identify its fraud category.")
iface.launch()
```

## Example Usage

Uncomment the example usage to test the function:

```python
# query = "A person received a call claiming to be from their bank and was asked to provide personal information to verify their account."
# result0, result1 = get_fraud_category(query, query_engine0, query_engine1)
# print("Result from Engine 0:", result0)
# print("Result from Engine 1:", result1)
```

## Directory Structure

Ensure your project directory is structured as follows:

```
/env/
    - your_pdfs.pdf
    - ...
your_script.py
```

## Running the Project

To run the project, simply execute your script:

```sh
python your_script.py
```

This will start the Gradio interface, where you can input descriptions to identify their fraud categories.

---
