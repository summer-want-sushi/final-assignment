from smolagents import Tool
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from wikipedia_utils import *
from youtube_utils import *


class MathModelQuerer(Tool):
    name = "math_model"
    description = "Solves advanced math problems using a pretrained\
    large language model specialized in mathematics. Ideal for symbolic reasoning, \
    calculus, algebra, and other technical math queries."

    inputs = {
        "problem": {
            "type": "string",
            "description": "Math problem to solve.",
        }
    }

    output_type = "string"

    def __init__(self, model_name="deepseek-ai/deepseek-math-7b-base"):
        print(f"Loading math model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("loaded tokenizer")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        print("loaded auto model")

        self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        print("loaded coonfig")

        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        print("loaded pad token")

    def forward(self, problem: str) -> str:
        try:
            print(f"[MathModelTool] Question: {problem}")

            inputs = self.tokenizer(problem, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=100)

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return result
        except:
            return f"Failed using the tool {self.name}"


class CodeModelQuerer(Tool):
    name = "code_querer"
    description = "Generates code snippets based on a natural language description of a\
    programming task using a powerful coding-focused language model. Suitable\
    for solving coding problems, generating functions, or implementing algorithms."

    inputs = {
        "problem": {
            "type": "string",
            "description": "Description of a code sample to be generated",
        }
    }

    output_type = "string"

    def __init__(self, model_name="Qwen/Qwen2.5-Coder-32B-Instruct"):
        from smolagents import HfApiModel
        print(f"Loading llm for Code tool: {model_name}")
        self.model = HfApiModel()

    def forward(self, problem: str) -> str:
        try:
            return self.model.generate(problem, max_new_tokens=512)
        except:
            return f"Failed using the tool {self.name}"


class WikipediaPageFetcher(Tool):
    name = "wiki_page_fetcher"
    description =' Searches and fetches summaries from Wikipedia for any topic,\
    across all supported languages and versions. Only a single query string is required as input.'



    inputs = {
        "query": {
            "type": "string",
            "description": "Topic of wikipedia search",
        }
    }

    output_type = "string"

    def forward(self, query: str) -> str:
        try:
            wiki_query = query(query)
            wiki_page = fetch_wikipedia_page(wiki_query)
            return wiki_page
        except:
            return f"Failed using the tool {self.name}"


class YoutubeTranscriptFetcher(Tool):
    name = "youtube_transcript_fetcher"
    description ="Fetches the English transcript of a YouTube video using either a direct video \
    ID or a URL that includes one. Accepts a query containing the link or the raw video ID directly. Returns the transcript as plain text."

    inputs = {
        "query": {
            "type": "string",
            "description": "A query that includes youtube id."
        },
        "video_id" : {
            "type" : "string",
            "description" : "Optional string with video id from youtube.",
            "nullable"  : True
        }
    }

    output_type = "string"

    def forward(self, query: str, video_id=None) -> str:
        try:
            if video_id is None:
                video_id = get_youtube_video_id(query)

            fetched_transcript = fetch_transcript_english(video_id)

            return post_process_transcript(fetched_transcript)
        except:
            return f"Failed using the tool {self.name}"
