"""Canvas course loader."""

import datetime
import os
from pathlib import Path

from langchain_community.document_loaders import (
    UnstructuredCSVLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    UnstructuredTSVLoader,
    UnstructuredXMLLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import tqdm

from .utils import extract_zip


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

# Skip binary formats that require heavy optional dependencies.
SKIP_EXTENSIONS = IMAGE_EXTENSIONS | {".ppt", ".pptx"}

FILE_LOADERS: dict[str, type[BaseLoader]] = {
    ".html": UnstructuredHTMLLoader,
    ".xml": UnstructuredXMLLoader,
    ".pdf": UnstructuredPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".csv": UnstructuredCSVLoader,
    ".tsv": UnstructuredTSVLoader,
}


class CanvasLoader(BaseLoader):
    """Load documents from a Canvas ``.imscc`` archive."""

    def __init__(self, file_path: str) -> None:
        """Create a loader for ``file_path``.

        Parameters
        ----------
        file_path:
            Path to the Canvas ``.imscc`` export.
        """
        path = Path(file_path)
        if not path.is_file():
            msg = f"Canvas file '{file_path}' does not exist or is not a file."
            raise FileNotFoundError(msg)
        self.zipped_file_path = str(path)
        self.course = path.stem

    def load(self) -> list[Document]:
        """Load all documents from the archive."""
        file_paths = extract_zip(self.zipped_file_path)
        return self._load_files(file_paths)

    def _load_files(self, list_of_files_to_load: list[str]) -> list[Document]:
        """
        Load the files from the list of files to load.

        Args:
            list_of_files_to_load (list): A list of file paths to load.

        Returns:
            list[Document]: A list of loaded documents.
        """
        loaded_documents: list[Document] = []
        for file_path in tqdm.tqdm(list_of_files_to_load):
            if not os.path.isfile(file_path):
                continue
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in SKIP_EXTENSIONS:
                continue  # Skip unsupported binary files

            loader_cls = FILE_LOADERS.get(file_extension)
            if loader_cls is not None:
                new_documents = loader_cls(file_path).load()  # type: ignore[call-arg]
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                    content = file.read()
                new_documents = [
                    Document(page_content=content, metadata={"source": file_path})
                ]

            timestamp = datetime.datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).isoformat()
            for doc in new_documents:
                doc.metadata.setdefault("source", file_path)
                doc.metadata["course"] = self.course
                doc.metadata["timestamp"] = timestamp
            loaded_documents += new_documents
        return loaded_documents


if __name__ == "__main__":
    # Example usage
    loader = CanvasLoader(
        "/Users/work/Downloads/special-topics-designing-and-deploying-ai-slash-ml-systems-export.imscc"
    )
    documents = loader.load()
    for doc in documents:
        print(doc)
