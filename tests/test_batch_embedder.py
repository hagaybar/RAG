import os
from scripts.api_clients.openai.batch_embedder import BatchEmbedder

# Sample test chunks
texts = [
    "Artificial intelligence is transforming industries.",
    "The capital of France is Paris.",
    "Quantum computing leverages the principles of quantum mechanics."
]

# Initialize embedder
embedder = BatchEmbedder(
    model="text-embedding-3-small",
    output_dir="dev/tests/batch_outputs",
    api_key=os.getenv("OPEN_AI")
)

# Run embedding batch
try:
    result = embedder.run(texts)
    print("\n✅ Batch Embedding Results:")
    for k, v in result.items():
        print(f"{k}: {str(v)[:60]}... [{len(v)} dimensions]")
except Exception as e:
    print(f"❌ Batch embedding failed: {e}")
