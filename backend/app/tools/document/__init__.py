"""Document tools for SMILE."""

from .save_document import save_document, DocumentSaveSchema
from .search_documents import search_documents, SearchDocumentsInput

__all__ = [
    'save_document',
    'DocumentSaveSchema',
    'search_documents',
    'SearchDocumentsInput'
] 