from typing import List, Dict, Any


class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # Deferred: langchain_text_splitters pulls sentence_transformers
        # (torch/sklearn/pandas) at import time, which would blow the
        # Render free 512MB budget during gunicorn boot.
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split pages into chunks while preserving page mapping.
        Returns: [{'content': str, 'page_number': int, 'file_name': str, 'chunk_index': int}, ...]
        """
        all_chunks = []
        chunk_index = 0
        
        for page in pages:
            chunks = self.splitter.split_text(page["content"])
            for chunk_text in chunks:
                if chunk_text.strip():
                    all_chunks.append({
                        "content": chunk_text,
                        "page_number": page["page_number"],
                        "file_name": page["file_name"],
                        "chunk_index": chunk_index,
                    })
                    chunk_index += 1
        
        return all_chunks