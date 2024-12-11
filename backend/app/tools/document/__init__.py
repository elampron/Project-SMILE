"""Document tools for SMILE."""

from .save_document import save_document, DocumentSaveSchema
from .search_documents import SearchDocumentsTool, SearchDocumentsInput

__all__ = [
    'save_document',
    'DocumentSaveSchema',
    'SearchDocumentsTool',
    'SearchDocumentsInput'
] 