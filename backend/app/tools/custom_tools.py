"""Custom tools for SMILE."""

# Import document tools
from .document import (
    save_document,
    DocumentSaveSchema,
    search_documents,
    SearchDocumentsInput
)

# Import system tools
from .system import (
    execute_python,
    execute_cmd,
    PythonExecuteSchema,
    CommandExecuteSchema
)

# Import entity tools
from .entity import (
    search_entities,
    SearchEntitiesInput
)

# Import memory tools
from .memory import (
    search_memories,
    SearchMemoriesInput
)

__all__ = [
    # Document tools
    'save_document',
    'DocumentSaveSchema',
    'search_documents',
    'SearchDocumentsInput',
    
    # System tools
    'execute_python',
    'execute_cmd',
    'PythonExecuteSchema',
    'CommandExecuteSchema',
    
    # Entity tools
    'search_entities',
    'SearchEntitiesInput',
    
    # Memory tools
    'search_memories',
    'SearchMemoriesInput'
]

