import json
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Literal,Dict, Any, Union
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PersonCategory(str, Enum):
    """Enumeration of possible person categories."""
    human = 'Human'
    pet = 'Pet'
    ai_agent = 'AI Agent'

class RelationshipType(str, Enum):
    """Enumeration of possible relationship types."""
    friend = 'Friend'
    family = 'Family'
    colleague = 'Colleague'
    pet_owner = 'Pet Owner'
    owns_pet = 'Owns Pet'
    member = 'Member'
    works_at = 'Works At'
    created_by = 'Created By'
    owns_ai_agent = 'Owns AI Agent'
    # Add other relationship types as needed


class BaseEntity(BaseModel):
    """
    Base model for all entities, including common fields.

    Attributes:
        id (UUID): Unique identifier for the entity.
        name (str): Name of the entity.
        type (str): Type of the entity (e.g., 'Person', 'Organization').
        created_at (datetime): Timestamp when the entity was created.
        updated_at (Optional[datetime]): Timestamp when the entity was last updated.
        embedding (Optional[List[float]]): Vector embedding for similarity search
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the entity.")
    db_id: Optional[str] = Field(None, description="Database ID of the entity.")
    name: str = Field(..., description="Name of the entity.")
    type: str = Field(..., description="Type of the entity (e.g., 'Person', 'Organization').")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the entity was created.")
    updated_at: Optional[datetime] = Field(None, description="Timestamp when the entity was last updated.")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding for similarity search")

class PersonEntity(BaseEntity):
    """
    Model representing a person, including humans, pets, and AI agents.

    Attributes:
        type (str): Type of the entity, fixed as 'Person'.
        category (PersonCategory): Category of the person (Human, Pet, AI Agent).
        nickname (Optional[str]): Optional nickname of the person.
        birth_date (Optional[str]): Date of birth in ISO 8601 format (YYYY-MM-DD).
        email (Optional[str]): Email address.
        phone (Optional[str]): Phone number.
        address (Optional[str]): Physical address.
        notes (Optional[str]): Additional notes about the person.
    """
    type: Literal["Person"] = Field(
        default="Person",
        description="Type of the entity, fixed as 'Person'."
    )
    category: PersonCategory = Field(..., description="Category of the person (Human, Pet, AI Agent).")
    nickname: Optional[str] = Field(None, description="Optional nickname of the person.")
    birth_date: Optional[str] = Field(None, description="Date of birth in ISO 8601 format (YYYY-MM-DD).")
    email: Optional[str] = Field('', description="Email address.")
    phone: Optional[str] = Field('', description="Phone number.")
    address: Optional[str] = Field('', description="Physical address.")
    notes: Optional[str] = Field('', description="Additional notes about the person.")

class OrganizationEntity(BaseEntity):
    """
    Model representing an organization or company.

    Attributes:
        type (str): Type of the entity, fixed as 'Organization'.
        industry (Optional[str]): Industry sector of the organization.
        website (Optional[str]): Website URL of the organization.
        address (Optional[str]): Physical address of the organization.
        notes (Optional[str]): Additional notes about the organization.
    """
    type: Literal["Organization"] = Field(
        default="Organization",
        description="Type of the entity, fixed as 'Organization'."
    )
    industry: Optional[str] = Field(None, description="Industry sector of the organization.")
    website: Optional[str] = Field(None, description="Website URL of the organization.")
    address: Optional[str] = Field(None, description="Physical address of the organization.")
    notes: Optional[str] = Field(None, description="Additional notes about the organization.")

class Relationship(BaseModel):
    """
    Model representing a relationship between two entities.

    Attributes:
        id (UUID): Unique identifier for the relationship.
        from_entity_id (UUID): UUID of the source entity.
        to_entity_id (UUID): UUID of the target entity.
        from_entity_db_id (Optional[str]): Database ID of the source entity.
        to_entity_db_id (Optional[str]): Database ID of the target entity.
        type (RelationshipType): Type of the relationship.
        since (Optional[str]): Date when the relationship started in ISO 8601 format.
        until (Optional[str]): Date when the relationship ended in ISO 8601 format.
        notes (Optional[str]): Additional notes about the relationship.
    """
    id: UUID = Field(default_factory=uuid4)
    from_entity_id: UUID
    to_entity_id: UUID
    from_entity_db_id: Optional[str] = Field(None, description="Database ID of the source entity.")
    to_entity_db_id: Optional[str] = Field(None, description="Database ID of the target entity.")
    type: RelationshipType = Field(..., description="Type of the relationship.")
    since: Optional[str] = Field(None, description="Date when the relationship started in ISO 8601 format (YYYY-MM-DD).")
    until: Optional[str] = Field(None, description="Date when the relationship ended in ISO 8601 format (YYYY-MM-DD).")
    notes: Optional[str] = Field(None, description="Additional notes about the relationship.")

class RelationshipResponse(BaseModel):
    from_entity_name: str
    to_entity_name: str
    type: RelationshipType
    since: Optional[str]
    until: Optional[str]
    notes: Optional[str]

class PersonResponse(BaseModel):
    """
    Model representing a person in the LLM response, excluding the 'id' field.

    Attributes:
        name (str): Name of the person.
        type (str): Type of the entity, fixed as 'Person'.
        category (PersonCategory): Category of the person (Human, Pet, AI Agent).
        nickname (Optional[str]): Optional nickname of the person.
        birth_date (Optional[str]): Date of birth in ISO 8601 format (YYYY-MM-DD).
        email (Optional[str]): Email address.
        phone (Optional[str]): Phone number.
        address (Optional[str]): Physical address.
        notes (Optional[str]): Additional notes about the person.
    """
    name: str = Field(..., description="Name of the person.")
    type: Literal["Person"] = Field(default="Person", description="Type of the entity, fixed as 'Person'.")
    category: PersonCategory = Field(..., description="Category of the person (Human, Pet, AI Agent).")
    nickname: Optional[str] = Field(None, description="Optional nickname of the person.")
    birth_date: Optional[str] = Field(None, description="Date of birth in ISO 8601 format (YYYY-MM-DD).")
    email: Optional[str] = Field(None, description="Email address.")
    phone: Optional[str] = Field(None, description="Phone number.")
    address: Optional[str] = Field(None, description="Physical address.")
    notes: Optional[str] = Field(None, description="Additional notes about the person.")

class OrganizationResponse(BaseModel):
    """
    Model representing an organization in the LLM response, excluding the 'id' field.

    Attributes:
        name (str): Name of the organization.
        type (str): Type of the entity, fixed as 'Organization'.
        industry (Optional[str]): Industry sector of the organization.
        website (Optional[str]): Website URL of the organization.
        address (Optional[str]): Physical address of the organization.
        notes (Optional[str]): Additional notes about the organization.
    """
    name: str = Field(..., description="Name of the organization.")
    type: Literal["Organization"] = Field(default="Organization", description="Type of the entity, fixed as 'Organization'.")
    industry: Optional[str] = Field(None, description="Industry sector of the organization.")
    website: Optional[str] = Field(None, description="Website URL of the organization.")
    address: Optional[str] = Field(None, description="Physical address of the organization.")
    notes: Optional[str] = Field(None, description="Additional notes about the organization.")

class RelationshipResponse(BaseModel):
    """
    Model representing a relationship in the LLM response, excluding the 'id' fields.

    Attributes:
        from_entity_name (str): Name of the source entity.
        to_entity_name (str): Name of the target entity.
        type (RelationshipType): Type of the relationship.
        since (Optional[str]): Date when the relationship started in ISO 8601 format (YYYY-MM-DD).
        until (Optional[str]): Date when the relationship ended in ISO 8601 format (YYYY-MM-DD).
        notes (Optional[str]): Additional notes about the relationship.
    """
    from_entity_name: str = Field(..., description="Name of the source entity.")
    to_entity_name: str = Field(..., description="Name of the target entity.")
    type: RelationshipType = Field(..., description="Type of the relationship.")
    since: Optional[str] = Field(None, description="Date when the relationship started in ISO 8601 format (YYYY-MM-DD).")
    until: Optional[str] = Field(None, description="Date when the relationship ended in ISO 8601 format (YYYY-MM-DD).")
    notes: Optional[str] = Field(None, description="Additional notes about the relationship.")


class EntityExtractorResponse(BaseModel):
    """
    Model representing the response from the entity extractor.

    Attributes:
        persons (List[PersonResponse]): List of person entities extracted.
        organizations (List[OrganizationResponse]): List of organization entities extracted.
        relationships (List[RelationshipResponse]): List of relationships extracted.
    """
    persons: List[PersonResponse] = Field(default_factory=list, description="List of person entities extracted.")
    organizations: List[OrganizationResponse] = Field(default_factory=list, description="List of organization entities extracted.")
    relationships: List[RelationshipResponse] = Field(default_factory=list, description="List of relationships extracted.")



class ActionItem(BaseModel):
    """
    Task or follow-up identified in conversation.
    Extract specific, actionable items with clear ownership and timing.

    Attributes:
        description (str): Clear, actionable task description
        assignee (Optional[str]): Person responsible for the task
        due_date (Optional[datetime]): Task deadline or target completion date
    """
    description: str
    assignee: Optional[str]
    due_date: Optional[datetime]

class Participant(BaseModel):
    """
    Person involved in or mentioned in conversation.
    Include both active participants and mentioned individuals.

    Attributes:
        name (str): Person's name as mentioned in conversation
        role (Optional[str]): Their role or relationship (e.g., "Team Lead", "Client")
    """
    name: str
    role: Optional[str]

class SentimentAnalysis(BaseModel):
    """
    Model representing the sentiment analysis of the conversation.

    Attributes:
        overall_sentiment (str): Overall sentiment (e.g., positive, negative, neutral).
        emotions (List[str]): Specific emotions expressed.
    """
    overall_sentiment: str
    emotions: List[str]

class TimeFrame(BaseModel):
    """
    Time context of discussed events or activities.
    Extract both explicit and implied timing information.
    Accepts flexible date formats including ISO strings and natural language.

    Attributes:
        start_time (Optional[datetime]): When events begin/began
        end_time (Optional[datetime]): When events end/ended
    """
    start_time: Optional[Union[datetime, str]] = Field(
        default_factory=datetime.utcnow,
        description="Start time of the timeframe. Can be datetime or string."
    )
    end_time: Optional[Union[datetime, str]] = Field(
        default_factory=datetime.utcnow,
        description="End time of the timeframe. Can be datetime or string."
    )

    @field_validator('start_time', 'end_time')
    @classmethod
    def parse_datetime(cls, value: Optional[Union[datetime, str]]) -> Optional[datetime]:
        """
        Validates and converts datetime fields.
        Accepts datetime objects and string representations.
        
        Args:
            value: datetime object or string representation of date/time
            
        Returns:
            datetime: Parsed datetime object
            
        Raises:
            ValueError: If string cannot be parsed into datetime
        """
        if value is None:
            return datetime.utcnow()
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                # Try parsing ISO format first
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # Fallback to basic format
                    return datetime.strptime(value, '%Y-%m-%d')
                except ValueError:
                    # If all parsing fails, return current time
                    logger.warning(f"Could not parse datetime string: {value}. Using current time.")
                    return datetime.utcnow()
        return datetime.utcnow()

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Custom serialization to ensure datetime objects are converted to ISO strings.
        """
        data = super().model_dump(*args, **kwargs)
        # Convert datetime objects to ISO format strings
        if data.get('start_time'):
            data['start_time'] = data['start_time'].isoformat()
        if data.get('end_time'):
            data['end_time'] = data['end_time'].isoformat()
        return data

class ConversationSummary(BaseModel):
    """
    Structured summary of conversation content and context.
    Extract key information into organized categories.

    Attributes:
        id (UUID): Unique identifier
        content (str): Clear summary of main points and decisions
        topics (List[str]): Key subjects discussed
        action_items (List[ActionItem]): Tasks and follow-ups
        participants (List[Participant]): People involved
        sentiments (Dict[str, Any]): Emotional tone analysis
        location (Optional[str]): Where conversation/events took place
        events (List[str]): Key events mentioned
        message_ids (List[str]): IDs of messages included in summary
        created_at (datetime): When summary was created
        start_time (Optional[datetime]): When events begin/began
        end_time (Optional[datetime]): When events end/ended
        embedding (Optional[List[float]]): Vector embedding for search
    """
    id: UUID = Field(default_factory=uuid4)
    content: str
    topics: List[str]
    action_items: List[ActionItem] = Field(default_factory=list)
    participants: List[Participant] = Field(default_factory=list)
    sentiments: Dict[str, Any] = Field(default_factory=dict)
    location: Optional[str] = None
    events: List[str] = Field(default_factory=list)
    message_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    start_time: Optional[datetime] = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = Field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding for similarity search")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            UUID: lambda v: str(v)
        }

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Custom serialization to handle datetime fields.
        """
        data = super().model_dump(*args, **kwargs)
        # Convert datetime objects to ISO format strings
        for field in ['created_at', 'start_time', 'end_time']:
            if data.get(field) and isinstance(data[field], datetime):
                data[field] = data[field].isoformat()
            elif data.get(field) and isinstance(data[field], str):
                # If it's already a string, verify it's in ISO format
                try:
                    datetime.fromisoformat(data[field].replace('Z', '+00:00'))
                except ValueError:
                    # If not in ISO format, set to current time
                    data[field] = datetime.utcnow().isoformat()
        return data

    @field_validator('start_time', 'end_time', 'created_at')
    @classmethod
    def validate_datetime(cls, value: Optional[Union[datetime, str]]) -> Optional[datetime]:
        """Validate and convert datetime fields."""
        if value is None:
            return datetime.utcnow()
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                return datetime.utcnow()
        return datetime.utcnow()

class Preference(BaseModel):
    """
    Model representing a user preference.
    
    Attributes:
        id (UUID): Unique identifier for the preference
        person_id (UUID): ID of the person this preference belongs to
        preference_type (str): Type of preference (e.g., 'organization', 'memory')
        importance (int): Importance level of the preference (1-5)
        details (Dict[str, Any]): Flexible dictionary for storing preference details
        created_at (datetime): Timestamp when the preference was created
        updated_at (datetime): Timestamp when the preference was last updated
        embedding (Optional[List[float]]): Vector embedding for similarity search
    """
    model_config = {
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "examples": [
                {
                    "details": {
                        "folder_name": "smiles_chest",
                        "reason": "needs good structure due to severe A.D.D."
                    }
                }
            ]
        }
    }
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    person_id: uuid.UUID
    preference_type: str
    importance: int = Field(ge=1, le=5)
    details: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding for similarity search")

class PreferenceResponse(BaseModel):
    """
    Model for preference response data.
    
    Attributes:
        person_name (str): Name of the person
        preference_type (str): Type of preference
        importance (int): Importance level (1-5)
        details (Dict[str, str]): Flexible dictionary for preference details
    """
    model_config = {
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "examples": [
                {
                    "details": {
                        "folder_name": "smiles_chest",
                        "reason": "needs good structure due to severe A.D.D."
                    }
                }
            ]
        }
    }
    
    person_name: str
    preference_type: str
    importance: int = Field(ge=1, le=5)
    details: Dict[str, str] = Field(default_factory=dict)

class PreferenceExtractorResponse(BaseModel):
    """
    Model for the complete preference extraction response.
    
    Attributes:
        preferences (List[PreferenceResponse]): List of extracted preferences
    """
    model_config = {
        "arbitrary_types_allowed": True
    }
    preferences: List[PreferenceResponse]

class Document(BaseModel):
    """
    Model representing a document stored in the library.
    
    Attributes:
        id (UUID): Unique identifier for the document
        name (str): Name of the document (including extension)
        content (str): Content of the document
        file_url (str): URL/path to the document file
        created_at (datetime): When the document was created
        updated_at (Optional[datetime]): When the document was last updated
        embedding (Optional[List[float]]): Vector embedding for similarity search
        metadata (Dict[str, Any]): Additional metadata about the document
    """
    id: UUID = Field(default_factory=uuid4)
    name: str
    content: str
    file_url: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding."""
        components = [
            self.name,
            self.content,
            " ".join(f"{k}:{v}" for k, v in self.metadata.items())
        ]
        return " ".join(filter(None, components))

class ConfidenceLevel(float, Enum):
    """Enumeration of confidence levels with semantic meaning."""
    CERTAIN = 1.0
    HIGHLY_LIKELY = 0.9
    LIKELY = 0.7
    POSSIBLE = 0.5
    UNCERTAIN = 0.3
    SPECULATIVE = 0.1

class MemorySource(str, Enum):
    """Sources of memory formation."""
    DIRECT_OBSERVATION = "direct_observation"  # From direct conversation
    INFERENCE = "inference"  # Derived from other memories
    USER_CONFIRMATION = "user_confirmation"  # Explicitly confirmed by user
    SYSTEM_ANALYSIS = "system_analysis"  # Generated by system analysis
    EXTERNAL_SOURCE = "external_source"  # From external data sources

class CognitiveAspect(str, Enum):
    """Different aspects of cognitive processing for the memory."""
    FACTUAL = "FACTUAL"  # Pure facts
    TEMPORAL = "TEMPORAL"  # Time-related information
    SPATIAL = "SPATIAL"  # Location-related
    CAUSAL = "CAUSAL"  # Cause-effect relationships
    EMOTIONAL = "EMOTIONAL"  # Emotional context
    BEHAVIORAL = "BEHAVIORAL"  # Behavior patterns
    SOCIAL = "SOCIAL"  # Social interactions
    PROCEDURAL = "PROCEDURAL"  # How-to knowledge

    @classmethod
    def _missing_(cls, value: str) -> Optional['CognitiveAspect']:
        """Handle case-insensitive enum values."""
        for member in cls:
            if member.value.upper() == value.upper():
                return member
        return None

class MemoryAssociation(BaseModel):
    """Represents connections between memories."""
    target_memory_id: UUID
    association_type: str
    strength: float = Field(ge=0.0, le=1.0)
    context: Optional[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TemporalContext(BaseModel):
    """Rich temporal information about the memory."""
    observed_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    valid_from: Optional[datetime] = Field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None
    is_recurring: bool = False
    recurrence_pattern: Optional[Dict[str, Any]] = None
    temporal_references: List[str] = Field(default_factory=list)
    
class SemanticAttributes(BaseModel):
    """Semantic enrichment of the memory."""
    keywords: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    sentiment: Optional[float] = Field(default=0.0)
    importance: float = Field(default=0.5)
    cognitive_aspects: List[CognitiveAspect] = Field(default_factory=lambda: [CognitiveAspect.FACTUAL])

class ValidationStatus(BaseModel):
    """Tracks the validation state of the memory."""
    is_valid: bool = True
    last_validated: datetime = Field(default_factory=datetime.utcnow)
    validation_source: MemorySource = Field(default=MemorySource.DIRECT_OBSERVATION)
    validation_notes: Optional[str] = None
    contradictions: List[UUID] = Field(default_factory=list)
    supporting_evidence: List[UUID] = Field(default_factory=list)

class MemoryRelations(BaseModel):
    """Tracks relationships with other system components."""
    related_entities: List[UUID] = Field(default_factory=list)
    related_summaries: List[UUID] = Field(default_factory=list)
    related_preferences: List[UUID] = Field(default_factory=list)
    associations: List[MemoryAssociation] = Field(default_factory=list)

class CognitiveMemory(BaseModel):
    """
    Core memory model representing a flexible, cognitive memory unit.
    
    This model is designed to capture any type of memory with rich metadata,
    semantic information, and cognitive aspects while maintaining flexibility
    in how information is stored and related.
    """
    # Core Identity
    id: UUID = Field(default_factory=uuid4)
    type: str  # Flexible type system
    sub_type: Optional[str] = None  # Optional refinement of type
    
    # Content
    content: str  # Primary textual content
    structured_data: Optional[Dict[str, Any]] = None  # Structured representation
    
    # Semantic and Cognitive Information
    semantic: SemanticAttributes = Field(default_factory=SemanticAttributes)
    embedding: Optional[List[float]] = None
    
    # Temporal and Source Information
    temporal: TemporalContext = Field(default_factory=TemporalContext)
    source: MemorySource = Field(default=MemorySource.DIRECT_OBSERVATION)
    source_messages: List[str] = Field(default_factory=list)
    
    # Confidence and Validation
    confidence: float = Field(default=1.0)
    validation: ValidationStatus = Field(default_factory=ValidationStatus)
    
    # Relations and Associations
    relations: MemoryRelations = Field(default_factory=MemoryRelations)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    # System Fields
    version: int = 1
    is_archived: bool = False
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    
    @model_validator(mode='after')
    def validate_dates(self) -> 'CognitiveMemory':
        """Ensure temporal consistency."""
        if self.temporal.valid_until and self.temporal.valid_from:
            if self.temporal.valid_until < self.temporal.valid_from:
                raise ValueError("valid_until must be after valid_from")
        return self

    def increment_access(self):
        """Update access metadata when memory is retrieved."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()

    def update_confidence(self, new_confidence: float, reason: str):
        """Update confidence with validation."""
        self.confidence = new_confidence
        self.updated_at = datetime.utcnow()
        # Could add confidence history tracking here

    def add_association(self, target_id: UUID, assoc_type: str, strength: float, context: Optional[str] = None):
        """Add a new association to another memory."""
        association = MemoryAssociation(
            target_memory_id=target_id,
            association_type=assoc_type,
            strength=strength,
            context=context
        )
        self.relations.associations.append(association)

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding."""
        components = [
            self.content,
            f"Type: {self.type}",
            f"Categories: {' '.join(self.semantic.categories)}",
            f"Keywords: {' '.join(self.semantic.keywords)}",
            json.dumps(self.structured_data) if self.structured_data else ""
        ]
        return " ".join(filter(None, components))

class MemoryIndex(BaseModel):
    """
    Tracks existing memory types and their statistics for the extractor.
    Helps maintain consistency in type usage.
    """
    type_counts: Dict[str, int] = Field(default_factory=dict)
    type_examples: Dict[str, List[str]] = Field(default_factory=lambda: {})
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def add_memory(self, memory: CognitiveMemory):
        """Update index with new memory."""
        self.type_counts[memory.type] = self.type_counts.get(memory.type, 0) + 1
        if memory.type not in self.type_examples:
            self.type_examples[memory.type] = []
        if len(self.type_examples[memory.type]) < 3:  # Keep up to 3 examples
            self.type_examples[memory.type].append(memory.content)
        self.last_updated = datetime.utcnow()

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Custom serialization to handle datetime fields.
        """
        data = super().model_dump(*args, **kwargs)
        # Convert datetime to ISO format string
        if data.get('last_updated'):
            data['last_updated'] = data['last_updated'].isoformat()
        return data

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }