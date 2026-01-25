from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from haystack import Pipeline, Document
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.generators.ollama import OllamaGenerator
from pydantic import BaseModel
from typing import Union
from unsloth import FastLanguageModel
import glob
import pandas as pd
import re

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ChrisMoe/Qwen3_segmented",
    max_seq_length=2048,
    load_in_4bit=True,
)

document_store = InMemoryDocumentStore()

template = """
Given only the following information, answer the question.
Ignore your own knowledge.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}?

IMPORTANT: Structure your answer consistently using this format:

1. Overall Pattern: [State the main grammatical pattern]

2. Component Breakdown:
   - List each component with its Chinese characters, pinyin (if relevant), and grammatical role
   - Use bullet points for each component

3. Pattern Structure: [Provide the general structure formula like "Subject + Verb + Object"]

4. Explanation: [Brief explanation of how this pattern works]

5. Segment-by-Segment Translation:
   - For each important segment, output:
     Chinese: <segment> | Pinyin: <pinyin> | English: <translation>
   - Use one bullet point per segment.
   - Keep segments aligned with how the sentence is naturally chunked.

Use this exact format for ALL sentences, whether simple or complex. Use English for explanation but use Chinese text for the terms being explained.
"""

pipe = Pipeline()
pipe.add_component('converter', TextFileToDocument())
pipe.add_component('cleaner', DocumentCleaner())
pipe.add_component('splitter', DocumentSplitter(split_by="sentence", split_length=5))
pipe.add_component(
    'writer',
    DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE),
)

pipe.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
pipe.add_component(
    "prompt_builder", PromptBuilder(template=template, required_variables=["query"])
)
pipe.add_component(
    "llm",
    OllamaGenerator(
        model="hopephoto/Qwen3-4B-Instruct-2507_q8:latest", url="http://localhost:11434"
    ),
)
pipe.connect('converter', 'cleaner')
pipe.connect("cleaner", "splitter")
pipe.connect("splitter", "writer")
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

sources = sorted(glob.glob("allset_grammar_txt/**/*.txt", recursive=True))
if not sources:
    sources = sorted(glob.glob("allset_grammar_txt/*.txt"))
if sources:
    print(f"Discovered {len(sources)} source files under allset_grammar_txt/")
else:
    print("Warning: no .txt source files found under allset_grammar_txt/")


class Sentence(BaseModel):
    content: str


app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def format_output(reply_text, query):
    """Format the LLM response for better readability."""
    # Remove list brackets and quotes if present
    if reply_text.startswith('[') and reply_text.endswith(']'):
        reply_text = reply_text.strip('[]').strip("'\"")

    # Clean up any extra quotes
    reply_text = reply_text.strip('"\'')

    # Build formatted output
    output = "\n" + "=" * 70 + "\n"
    output += "GRAMMAR PATTERN ANALYSIS\n"
    output += "=" * 70 + "\n\n"
    output += f"Query: {query}\n\n"
    output += "-" * 70 + "\n\n"
    output += reply_text + "\n\n"
    output += "=" * 70 + "\n"

    return output


@app.get("/")
def read_root():
    return {"message": "hello world!"}


@app.post("/api/v1/analyze")
def analyze(sentence: Sentence):
    response = pipe.run(
        {
            "converter": {"sources": sources},
            "prompt_builder": {"query": sentence.content},
            "retriever": {"query": sentence.content},
        }
    )

    def extract_reply_text(resp):
        # pipeline returns replies as list; pick first reply string
        try:
            replies = resp["llm"]["replies"]
            if isinstance(replies, (list, tuple)):
                return replies[0]
            return replies
        except Exception:
            return json.dumps(resp, ensure_ascii=False)

    reply_text = extract_reply_text(response)
    formatted_output = format_output(reply_text, sentence.content)

    return {"report": formatted_output}


@app.post("/api/v1/segment")
def segment_sentence(sentence: Sentence):
    messages = [
        {
            "role": "user",
            "content": "请将以下句子进行中文分词，如果你在句子里见到名字，不要分词",
        },
        {"role": "user", "content": sentence},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    attention_mask = inputs != tokenizer.pad_token_id
    outputs = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        max_new_tokens=128,
        min_new_tokens=1,
        do_sample=False,
        temperature=0.7,
        min_p=0.0,
        top_p=0.8,
        top_k=20,
    )

    generated_tokens = outputs[0][inputs.shape[1] :]
    segmented = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    segmented = re.sub(r"<\|.*?\|>", "", segmented)
    segmented = re.sub(r"</?\w+>", "", segmented)
    segmented = re.sub(
        r"^(assistant|user|system)\s*", "", segmented.strip(), flags=re.IGNORECASE
    )
    segmented = re.sub(r"请.*?分词[:：\-]+", "", segmented)
    segmented = re.sub(r"\s+", " ", segmented).strip()
    segmented = re.sub(r"分词结果[:：]\s*", "", segmented)
    segmented = re.sub(r"该句子.*?为[:：]\s*", "", segmented)
    segmented = re.sub(r"^输出结果[:：]\s*", "", segmented)

    print("→", segmented)

    return {"segments": segmented}
