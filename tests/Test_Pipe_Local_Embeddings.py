import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.pipeline.rag_pipeline import RAGPipeline as RAGPipeline


def test_extract_and_embed(config_file):
    pipeline = RAGPipeline(config_path=config_file)
    pipeline.load_config()
    pipeline.add_step("extract_and_chunk")
    pipeline.add_step("embed_chunks")
    pipeline.add_step("get_user_query", query="Please check any complaints related to Alma Analytics. What kind of complaints? what were the solutions given?")
    pipeline.add_step("retrieve", force=True)
    pipeline.add_step("generate_answer")
    pipeline.pipe_review()
    pipeline.run_steps()
    
if __name__ == "__main__":
    test_extract_and_embed(config_file="C:/Users/hagaybar/OneDrive - Tel-Aviv University/My Personal files/systems/Rag_Project/configs/tasks/test_full_local.yaml")
    # test_single_step_pipe()
