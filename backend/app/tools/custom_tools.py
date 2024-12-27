"""Custom tools for SMILE."""

# Import document tools
from .document import (
    save_document,
    DocumentSaveSchema,
    SearchDocumentsTool,
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
    SearchEntitiesTool,
    SearchEntitiesInput
)

# Import memory tools
from .memory import (
    SearchMemoriesTool,
    SearchMemoriesInput,
    SaveMemoryTool,
    SaveMemoryInput
)

# Import knowledge tools
from .knowledge import (
    SearchKnowledgeTool,
    SearchKnowledgeInput
)

__all__ = [
    # Document tools
    'save_document',
    'DocumentSaveSchema',
    'SearchDocumentsTool',
    'SearchDocumentsInput',
    
    # System tools
    'execute_python',
    'execute_cmd',
    'PythonExecuteSchema',
    'CommandExecuteSchema',
    
    # Entity tools
    'SearchEntitiesTool',
    'SearchEntitiesInput',
    
    # Memory tools
    'SearchMemoriesTool',
    'SearchMemoriesInput',
    'SaveMemoryTool',
    'SaveMemoryInput',
    
    # Knowledge tools
    'SearchKnowledgeTool',
    'SearchKnowledgeInput'
]

