import logging
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader

from typing import Iterator, Optional

logger = logging.getLogger(__name__)


class Page:
    """
    Represents a single page of a document.
    """

    def __init__(self, page_number:int, content: str, metadata: Optional[dict] = None):
        self.page_number = page_number
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Page(content={self.content[:30]}...)"  # Show first 30 characters for brevity


class DocumentLoaderPDF:
    """
    Document loader for PDF files.
    """

    def __init__(self, file_path: Path):
        self._file_path = file_path
        self._py_my_pdf_loader = None

    async def initialize(self) -> None:
        """
        Initialize the PDF document loader.

        :return: None
        :raises DocLoaderInitializationError: If the PDF document loader fails to initialize.
        """

        try:
            self._py_my_pdf_loader = PyMuPDFLoader(self._file_path)
        except Exception as e:
            raise Exception("Failed to initialize PDF document loader") from e

    async def deinitialize(self) -> None:
        """
        Deinitialize the PDF document loader.

        :return: None
        """
        # No specific deinitialization needed for PyMuPDFLoader
        pass

    def get_pages_itr(self) -> Iterator[Page]:
        """
        Asynchronously retrieve all pages of the PDF document.

        :returns: An async iterator yielding Page objects.
        :raises DocLoaderRuntimeError: If an error occurs while retrieving pages.
        """

        if not self._py_my_pdf_loader:
            raise Exception("PDF document loader is not initialized")

        current_page = 1

        try:
            for document in self._py_my_pdf_loader.lazy_load():
                yield Page(page_number=current_page, content=document.page_content, metadata=document.metadata)
                current_page += 1
        except Exception as e:
            raise Exception("Failed to retrieve pages from PDF document") from e
