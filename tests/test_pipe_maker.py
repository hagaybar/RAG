import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from scripts.pipe_maker.rag_pipeline import RAGPipeline as RAGPipeline

def test_single_step_pipe():
    # 1. Create the pipeline
    pipeline = RAGPipeline()

    # 2. Create a config with real values
    config_path = pipeline.configure_task(
        task_name="test_extract_chunk_embedding",
        overrides={
           "embedding": {
                "mode": "local",
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dim": 384,
                "output_dir": "C:/Users/hagaybar/OneDrive - Tel-Aviv University/My Personal files/systems/AI Project/embeddings/emails/local_embeddings/debug_test",
                "index_filename": "debug.index",
                "metadata_filename": "debug.tsv"
            },

            "paths" : {
                  "chunked_emails": "C:/Users/hagaybar/OneDrive - Tel-Aviv University/My Personal files/systems/AI Project/DATA/emails/chunks/debug_test/chunked_emails.tsv"
            },
            "outlook": {
                "account_name": "hagaybar@tauex.tau.ac.il",
                "folder_path": "LISTS>ALMA L",
                "days_to_fetch": 2
            }
        }
    )




    # 3. Load the config
    pipeline.load_config(config_path)

    # 4. View the config
    pipeline.pipe_review()

    # 5. Add one step
    pipeline.add_step("extract_and_chunk")
    pipeline.add_step("embed_chunks")

    # 6. Run just that one step
    pipeline.run_steps()

def test_diplay_pipe_and_steps(config_file):
    pipeline = RAGPipeline(config_path=config_file)
    pipeline.load_config(config_file)
    pipeline.add_step("get_user_query", query="Are there known issues with auto loan renewal rules in Alma?")
    pipeline.add_step("retrieve", force=True)
    pipeline.add_step("generate_answer")  # No args needed
    pipeline.run_steps()
    print("---------------------")

    print(pipeline.last_chunks["context"])



    
if __name__ == "__main__":
    test_diplay_pipe_and_steps(config_file="C:/Users/hagaybar/OneDrive - Tel-Aviv University/My Personal files/systems/AI Project/task_configs/test_full_local.yaml")
    # test_single_step_pipe()
