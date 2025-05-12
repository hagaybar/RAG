import os
from pathlib import Path
from typing import Optional
import pandas as pd
import yaml
from datetime import datetime


from scripts.data_processing.email.config_loader import ConfigLoader
from scripts.data_processing.email.email_fetcher import EmailFetcher

from scripts.chunking.text_chunker_v2 import TextChunker
from scripts.retrieval.chunk_retriever_v3 import ChunkRetriever
from scripts.prompting.prompt_builder import EmailPromptBuilder
from scripts.api_clients.openai.gptApiClient import APIClient

from scripts.utils.config_templates import get_default_config
from scripts.utils.merge_utils import deep_merge
from scripts.utils.logger import LoggerManager

from scripts.utils.yaml_utils import QuotedStringDumper


class RAGPipeline:
    STEP_DEPENDENCIES = {
    "extract_and_chunk": [],
    "embed_chunks": [],
    "get_user_query": [],
    "retrieve": ["embed_chunks", "get_user_query"],
    "generate_answer": ["retrieve"]

}

    def __init__(self, config_path: Optional[str] = None):
        self.logger = LoggerManager.get_logger("RAGPipeline")
        self.logger.info("Initializing RAGPipeline...")
        self.config_loader = None
        self.config = None
        self.config_path = None
        self.mode = None
        self.embedder = None
        self.retriever = None
        self.chunked_file = None
        self.index_path = None
        self.metadata_path = None
        self.steps = []  # list of (step_name, kwargs)
        self.query = None
        self.last_chunks = None  #

        if config_path:
            self.config_path = config_path
            self.load_config(config_path)

    def load_config(self, path: str = None) -> None:
        self.logger.info("Loading configuration...")
        if path is None:
            path = self.config_path
        self.config_loader = ConfigLoader(path)
        self.config = self.config_loader._load_config()
        self.validate_config()
        self.logger.info(f"Configuration loaded successfully from: {path}")
        self.mode = self.config["embedding"]["mode"]
        output_dir = self.config["embedding"]["output_dir"]
        self.index_path = os.path.join(output_dir, self.config["embedding"]["index_filename"])
        self.metadata_path = os.path.join(output_dir, self.config["embedding"]["metadata_filename"])


        self.embedder = self._create_embedder()

    def ensure_config_loaded(self):
        if not self.config:
            raise RuntimeError("No configuration loaded. Please call load_config(path) first.")

    def get_user_query(self, query: str):
        """
        Set the user query for downstream use in retrieval and generation steps.
        """
        self.logger.info("Setting user query...")
        self.query = query
        print(f"🔍 Query set: {query}")
        self.logger.info(f"User query set: {query}")

    def _create_embedder(self):
        self.logger.info("Creating embedder...")
        if not self.config:
            self.logger.error("Config not loaded. Cannot create embedder.")
            raise RuntimeError("Config not loaded. Cannot create embedder.")

        mode = self.config["embedding"]["mode"]
        model_name = self.config["embedding"]["model_name"]
        embedding_dim = self.config["embedding"]["embedding_dim"]
        output_dir = self.config["embedding"]["output_dir"]
        index_filename = self.config["embedding"]["index_filename"]
        metadata_filename = self.config["embedding"]["metadata_filename"]

        if mode == "local":
            from scripts.embedding.local_model_embedder import LocalModelEmbedder
            from scripts.embedding.general_purpose_embedder import GeneralPurposeEmbedder

            embedder_client = LocalModelEmbedder(model_name)
            self.logger.info("Local model embedder created.")
            return GeneralPurposeEmbedder(
                embedder_client=embedder_client,
                embedding_dim=embedding_dim,
                output_dir=output_dir,
                index_filename=index_filename,
                metadata_filename=metadata_filename
            )

        elif mode == "api":
            from scripts.api_clients.openai.gptApiClient import APIClient
            from scripts.embedding.general_purpose_embedder import GeneralPurposeEmbedder

            api_client = APIClient(config=self.config)
            self.logger.info("API client created.")
            return GeneralPurposeEmbedder(
                embedder_client=api_client,
                embedding_dim=embedding_dim,
                output_dir=output_dir,
                index_filename=index_filename,
                metadata_filename=metadata_filename
            )

        else:
            raise ValueError(f"Unsupported embedding mode: {mode}")

    def extract_and_chunk(self) -> str:
        self.logger.info("Starting email extraction and chunking...")
        self.ensure_config_loaded()

        # ✅ Step 1: Fetch emails using the EmailFetcher class
        fetcher = EmailFetcher(self.config)
        # tsv_path = fetcher.fetch_emails_from_folder(return_dataframe=False)
        raw_emails_df = fetcher.fetch_emails_from_folder(return_dataframe=True)


        # ✅ Step 2: check if the DataFrame is empty
        if raw_emails_df.empty or raw_emails_df.columns.empty:
            raise ValueError("❌ No emails fetched — DataFrame is empty. Check Outlook folder path or email filtering.")


        # ✅ Step 3: Chunk based on "Cleaned Body" (assumes it's already cleaned)
        output_file = self.config["paths"]["chunked_emails"]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print("📦 raw_emails preview:")
        print(raw_emails_df.head(3).to_string())


        chunk_cfg = self.config["chunking"]
        chunker = TextChunker(
            max_chunk_size=chunk_cfg.get("max_chunk_size", 500),
            overlap=chunk_cfg.get("overlap", 50),
            min_chunk_size=chunk_cfg.get("min_chunk_size", 150),
            similarity_threshold=chunk_cfg.get("similarity_threshold", 0.8),
            language_model=chunk_cfg.get("language_model", "en_core_web_sm"),
            embedding_model=chunk_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )

        df = pd.DataFrame(raw_emails_df)
        df["Chunks"] = df["Cleaned Body"].apply(lambda x: chunker.chunk_text(str(x)))
        df_chunks = df.explode("Chunks").reset_index(drop=True).rename(columns={"Chunks": "Chunk"})
        df_chunks.to_csv(output_file, sep="\t", index=False)

        self.chunked_file = output_file
        print(f"✅ Chunked email data saved to: {output_file}")
        self.logger.info(f"Chunked email data saved to: {output_file}")
        return output_file

    def embed_chunks(self) -> None:
        self.ensure_config_loaded()
        self.logger.info("Starting chunk embedding...")
        if not self.chunked_file:
            self.logger.error("No chunked file available. Run extract_and_chunk() first.")
            raise RuntimeError("No chunked file available. Run extract_and_chunk() first.")
        self.embedder.run(self.chunked_file, text_column="Chunk")


    def retrieve(self, query: Optional[str] = None) -> dict:
        self.logger.info("Starting chunk retrieval...")
        self.ensure_config_loaded()

        top_k = self.config.get("retrieval", {}).get("top_k", 5)  # ← Read from config

        retriever = ChunkRetriever(
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            top_k=top_k,
            config=self.config
        )

        query = query or self.query
        if not query:
            raise ValueError("No query provided. Use get_user_query() or pass query explicitly.")

        query_vector = self.embedder.embed_query(query)
        result = retriever.retrieve(query_vector=query_vector)
        self.last_chunks = result

        self.logger.info(f"Retrieved {len(result['context'])} relevant chunks for query: {query}")
        return result


    def retrieve_old(self, query: Optional[str] = None) -> dict:
        self.logger.info("Starting chunk retrieval...")
        self.ensure_config_loaded()
        retriever = ChunkRetriever(index_path=self.index_path, metadata_path=self.metadata_path)
        query = query or self.query
        if not query:
            raise ValueError("No query provided. Use get_user_query() or pass query explicitly.")
        query_vector = self.embedder.embed_query(query)
        result = retriever.retrieve(query_vector=query_vector)
        self.last_chunks = result
        self.logger.info(f"Retrieved {len(result['context'])} relevant chunks for query: {query}")
        return result

    def generate_answer(self, query: Optional[str] = None, chunks: Optional[dict] = None) -> str:
        self.logger.info("Generating answer...")
        self.ensure_config_loaded()

        query = query or self.query
        if not query:
            self.logger.error("No query provided. Use get_user_query() before calling generate_answer().")
            raise ValueError("No query provided. Use get_user_query() before calling generate_answer().")

        chunks = chunks or getattr(self, "last_chunks", None)
        if not chunks:
            self.logger.error("No chunks provided. Ensure retrieve() ran before generate_answer().")
            raise ValueError("No chunks provided. Ensure retrieve() ran before generate_answer().")

        self.logger.info("Building prompt...")
        prompt_builder = EmailPromptBuilder()
        client = APIClient(config=self.config)
        prompt = prompt_builder.build(query, chunks["context"])
        self.logger.info(f"Prompt being sent:\n{prompt}")
        # print(f"🧠 Prompt being sent:\n{prompt}")
        answer = client.send_completion_request(prompt)
        self.logger.info(f"Generated answer:\n{answer}")

        print("💬 Generated Answer:\n", answer)
        return answer


    def run_full_pipeline(self, query: str) -> str:
        self.ensure_config_loaded()
        self.extract_and_chunk()
        self.embed_chunks()
        chunks = self.retrieve(query)
        answer = self.generate_answer(query, chunks)
        return answer

    def configure_task(self, task_name: str, output_format: str = "yaml", overrides: Optional[dict] = None) -> str:
        if self.config is None:
            self.config = {}

        default_config = get_default_config(task_name)
        task_config = deep_merge(default_config, overrides or {})

        os.makedirs("task_configs", exist_ok=True)
        save_path = os.path.join("task_configs", f"{task_name}.{output_format}")

        if output_format == "yaml":
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    task_config,
                    f,
                    Dumper=QuotedStringDumper,
                    default_flow_style=False
                )
        elif output_format == "json":
            import json
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(task_config, f, indent=4)
        else:
            raise ValueError("Output format must be 'yaml' or 'json'.")

        print(f"✅ Task configuration saved to: {save_path}")
        return save_path

    def validate_config(self):
        required = [
            ("embedding", dict),
            ("embedding.model_name", str),
            ("embedding.output_dir", str),
            ("embedding.embedding_dim", int),
            ("retrieval.top_k", int),
            ("outlook.account_name", str),
            ("outlook.folder_path", str),
            ("outlook.days_to_fetch", int),
        ]

        for key_path, expected_type in required:
            parts = key_path.split(".")
            value = self.config
            for part in parts:
                if part not in value:
                    raise KeyError(f"Missing required config key: {key_path}")
                value = value[part]
            if not isinstance(value, expected_type):
                self.logger.error(f"Config key {key_path} must be of type {expected_type.__name__}, got {type(value).__name__}")
                raise TypeError(f"Config key {key_path} must be of type {expected_type.__name__}, got {type(value).__name__}")
            else:
                pass
                # print(f"Config key {key_path} is valid.")
                # self.logger.info(f"Config key {key_path} is valid.")

    def add_step(self, step_name: str, force: bool = False, **kwargs):
        """
        Add a pipeline step by name, along with optional parameters.

        Args:
            step_name (str): Name of the pipeline step to add.
            force (bool): If True, bypasses dependency checks. Use with caution.
            **kwargs: Optional arguments to pass to the step during execution.

        Raises:
            AttributeError: If the step is not a method of RAGPipeline.
            ValueError: If dependencies are not met and force=False.
        """
        if not hasattr(self, step_name):
            raise AttributeError(f"Step '{step_name}' is not a method of RAGPipeline.")

        if step_name not in self.STEP_DEPENDENCIES:
            raise ValueError(f"Step '{step_name}' is not a recognized pipeline step.")

        added_steps = [s for s, _ in self.steps]

        # Check for missing dependencies
        missing_dependencies = [
            dep for dep in self.STEP_DEPENDENCIES[step_name]
            if dep not in added_steps
        ]

        if missing_dependencies and not force:
            raise ValueError(
                f"Cannot add step '{step_name}' — missing prerequisite(s): {missing_dependencies}.\n"
                f"You can override this check with force=True if you are sure these steps were already completed."
            )

        self.steps.append((step_name, kwargs))
        print(f"✅ Step '{step_name}' added{' (force override)' if force else ''}.")

    def clear_steps(self):
        """Remove all configured steps from the pipeline."""
        self.steps.clear()
        print("🧹 Pipeline steps cleared.")

    def run_steps(self):
        """
        Execute all steps added via `add_step()` in order.
        """
        print("🚀 Running configured pipeline steps...\n")
        for step_name, kwargs in self.steps:
            step_fn = getattr(self, step_name)
            print(f"➡️ Running step: {step_name}")
            try:
                result = step_fn(**kwargs)
                print(f"✅ Step '{step_name}' completed.\n")

                # If final step is generate_answer → show + save result
                if step_name == "generate_answer":
                    print("🧠 Final Answer:")
                    print(result)

                    task_name = self.config.get("task_name", "unnamed_task")
                    output_path = os.path.join("outputs", "answers", f"{task_name}.txt")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(result)
                    print(f"📁 Answer saved to: {output_path}")

            except Exception as e:
                print(f"❌ Step '{step_name}' failed with error: {e}")
                raise


    def pipe_review(self):
        """
        Display a review of the current pipeline configuration, including steps, model, and paths.
        """
        self.ensure_config_loaded()

        print("\n🚀 RAG Pipeline Configuration Review")
        print("==========================================")
        print(f"Config Source: {getattr(self.config_loader, 'config_path', '[Not loaded via config_loader]')}\n")

        print("[Steps in Pipeline]")
        if self.steps:
            for i, (step, kwargs) in enumerate(self.steps, start=1):
                print(f"  {i}. {step} {'(with arguments)' if kwargs else ''}")
        else:
            print("  (No steps added to pipeline yet)")
        print()

        emb_cfg = self.config.get("embedding", {})
        print("[Chunking]")
        print(f"Max Chunk Size: {self.config['chunking']['max_chunk_size']}")
        print(f"Overlap: {self.config['chunking']['overlap']}")
        print(f"Min Chunk Size: {self.config['chunking']['min_chunk_size']}")
        print(f"Similarity Threshold: {self.config['chunking']['similarity_threshold']}")
        print(f"Language Model: {self.config['chunking']['language_model']}")
        print(f"Embedding Model for Similarity: {self.config['chunking']['embedding_model']}\n")

        print("[Embedding]")
        print(f"  Mode: {emb_cfg.get('mode', 'N/A')}")
        print(f"  Model: {emb_cfg.get('model_name', 'N/A')}")
        print(f"  Embedding Dimension: {emb_cfg.get('embedding_dim', 'N/A')}")
        print(f"  Output Directory: {emb_cfg.get('output_dir', 'N/A')}\n")

        print("[Retrieval]")
        print(f"  FAISS Index Path: {getattr(self, 'index_path', 'N/A')}")
        print(f"  Metadata Path: {getattr(self, 'metadata_path', 'N/A')}")
        print(f"  Top-K: {self.config.get('retrieval', {}).get('top_k', 'N/A')}\n")

        print("[Prompting]")
        print("  Prompt Builder: EmailPromptBuilder (default)\n")

        print("[Answer Generation]")
        gen_cfg = self.config.get("generation", {})
        print(f"  Model: {gen_cfg.get('model', 'openai-gpt-4')}")
        print("==========================================\n")

    def describe_steps(self):
        """
        Print a list of available pipeline steps, their descriptions, and dependencies.
        """
        STEP_INFO = {
            "extract_and_chunk": {
                "desc": "Fetch emails from Outlook and chunk them into segments.",
                "depends_on": []
            },
            "embed_chunks": {
                "desc": "Embed chunked text using a local model or OpenAI API.",
                "depends_on": []
            },
            "get_user_query": {
                "desc": "Prompt the user (or system) to input a natural language query.",
                "depends_on": []
            },
            "retrieve": {
                "desc": "Use the embedded query to retrieve top-K relevant chunks.",
                "depends_on": ["embed_chunks", "get_user_query"]
            },
            "generate_answer": {
                "desc": "Generate a natural language answer using retrieved chunks.",
                "depends_on": ["retrieve"]
            },
        }

        print("\n📚 Available RAG Pipeline Steps")
        print("=======================================")
        for step, info in STEP_INFO.items():
            deps = ", ".join(info["depends_on"]) if info["depends_on"] else "None"
            print(f"🔹 {step}")
            print(f"    Description: {info['desc']}")
            print(f"    Depends on: {deps}\n")
