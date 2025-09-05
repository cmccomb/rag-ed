"""Piazza course loader."""

import datetime
import os
from pathlib import Path

import langchain_community.document_loaders
import langchain_core.document_loaders
import langchain_core.documents
import tqdm

from .utils import extract_zip


class PiazzaLoader(langchain_core.document_loaders.BaseLoader):
    """Load documents from a Piazza export archive."""

    def __init__(self, file_path: str) -> None:
        """Create a loader for ``file_path``.

        Parameters
        ----------
        file_path:
            Path to the Piazza ``.zip`` export.
        """
        path = Path(file_path)
        if not path.is_file():
            msg = f"Piazza file '{file_path}' does not exist or is not a file."
            raise FileNotFoundError(msg)
        self.zipped_file_path = str(path)
        self.course = path.stem

    def load(self) -> list[langchain_core.documents.Document]:
        """Load all documents from the archive."""
        file_paths = extract_zip(self.zipped_file_path)
        return self._load_files(file_paths)

    def _load_files(
        self, list_of_files_to_load: list[str]
    ) -> list[langchain_core.documents.Document]:
        """
        Load the files from the list of files to load.

        Args:
            list_of_files_to_load (list): A list of file paths to load.

        Returns:
            list[Document]: A list of loaded documents.
        """
        loaded_documents = []
        for file_path in tqdm.tqdm(list_of_files_to_load):
            if os.path.isfile(file_path):
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension == ".csv":
                    new_documents = langchain_community.document_loaders.CSVLoader(
                        file_path
                    ).load()
                elif file_extension == ".json":
                    new_documents = langchain_community.document_loaders.JSONLoader(
                        file_path, jq_schema=".", text_content=False
                    ).load()
                else:
                    continue  # Skip other file types

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
    loader = PiazzaLoader("/Users/work/Downloads/mech2-piazza.zip")
    documents = loader.load()
    for doc in documents:
        print(doc)
