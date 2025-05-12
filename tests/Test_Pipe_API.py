import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.pipeline.rag_pipeline import RAGPipeline as RAGPipeline

def create_pipeline_task():
    pipeline = RAGPipeline()
    pipeline.configure_task(task_name="test_full_api",
                            output_format="yaml",overrides={
                                "embedding": {
                                    "mode": "api",
                                    "model_name": "text-embedding-3-small",
                                    "embedding_dim": 1536,
                                    "output_dir": "C:/Users/hagaybar/OneDrive - Tel-Aviv University/My Personal files/systems/Rag_Project/data/embeddings/emails_api/debug_test",
                                    "index_filename": "api_test.index",
                                    "metadata_filename": "api_test.tsv"
                                },
                                "paths": {
                                    "email_dir": "data/cleaned",
                                    "output_file": "emails_to_api.tsv",
                                    "chunked_emails": "C:/Users/hagaybar/OneDrive - Tel-Aviv University/My Personal files/systems/Rag_Project/data/chunks/debug_test/chunked_emails_api.tsv",
                                    "log_dir": "C:/Users/hagaybar/OneDrive - Tel-Aviv University/My Personal files/systems/Rag_Project/data/logs"
                                },
                                "outlook": {
                                    "account_name": "LibraryHelpdesk",
                                    "folder_path": "Inbox>AlmaSupport",
                                    "days_to_fetch": 3
                                },
                                "retrieval": {
                                    "top_k": 8
                                },
                                "chunking": {
                                    "max_chunk_size": 450,
                                    "overlap": 50,
                                    "min_chunk_size": 150,
                                    "similarity_threshold": 0.8,
                                    "language_model": "en_core_web_sm",
                                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
                                },
                                "generation": {
                                    "model": "gpt-4o-mini"
                                }

                            })

def review_pipeline_task():
    pipeline = RAGPipeline(config_path=r"configs\tasks\test_full_api.yaml")
    pipeline.load_config()
    pipeline.pipe_review()


def test_extract_and_embed(config_file):
    pipeline = RAGPipeline(config_path=config_file)
    pipeline.load_config()
    pipeline.add_step("extract_and_chunk")
    pipeline.add_step("embed_chunks")
    pipeline.add_step("get_user_query", query="Please check if there is a problem of emails from Alma going into SPAM folders or even blocked?")
    pipeline.add_step("retrieve", force=True)
    pipeline.add_step("generate_answer")
    pipeline.pipe_review()
    pipeline.run_steps()
    
if __name__ == "__main__":
    test_extract_and_embed(config_file=r"configs\tasks\test_full_api.yaml")
    