# Memories extracting
# This is where text files (.txt) are read and converted into sentences, chunks, and sfts suitable for LLM training.

The file: memories_batch_pipeline.py takes a <input> and <output> directories.  The <input> directory consists of texts files. The <output> directory will be created and contains:

1. manifest.json
2. <sentences>
3. <chunks>
4. <sft>
