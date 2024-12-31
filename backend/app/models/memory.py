import json
import uuid
from app.services.neo4j.utils import Neo4jEncoder
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
    Model for preference response data. This model captures individual preferences expressed in conversations,
    including who the preference belongs to, what type of preference it is, and detailed information about the preference.
    
    Attributes:
        person_name (str): The full name of the person the preference belongs to. Use 'Eric Lampron' for the user,
            'Smiles' for the AI assistant, or the actual name for other people mentioned.
        preference_type (str): Category or domain of the preference (e.g., 'communication', 'system', 'food', 'travel', 'location').
            Can be either an existing type or a new meaningful category.
        importance (int): Priority level from 1 to 5, where:
            1 = Nice to have
            2 = Somewhat important
            3 = Important
            4 = Very important
            5 = Critical (ONLY use when explicitly stated as critical/essential)
        details (Dict[str, str]): Structured information about the preference, including:
            - Specific values or choices (e.g., {"theme": "dark"})
            - Reasons or context (e.g., {"reason": "easier on eyes"})
            - Related attributes (e.g., {"frequency": "daily", "time": "morning"})
    """
    model_config = {
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "examples": [
                {
                    "person_name": "Eric Lampron",
                    "preference_type": "location",
                    "importance": 4,
                    "details": {
                        "city": "Montreal",
                        "country": "Canada",
                        "reason": "current residence",
                        "context": "used for weather and local services"
                    }
                },
                {
                    "person_name": "Eric Lampron",
                    "preference_type": "family",
                    "importance": 4,
                    "details": {
                        "family_members": "son Arthur (3yo), nephew Ezequiel (2yo)",
                        "birthdays": "Arthur: December 17, Ezequiel: December 31",
                        "context": "celebrating family birthdays together"
                    }
                }
            ]
        }
    }
    
    person_name: str = Field(
        ...,
        description="Full name of the person. Use 'Eric Lampron' for user, 'Smiles' for AI, or actual name for others"
    )
    preference_type: str = Field(
        ...,
        description="Category of the preference (e.g., 'communication', 'system', 'food', 'travel', 'location')"
    )
    importance: int = Field(
        ge=1,
        le=5,
        description="Importance level (1-5). Use 5 ONLY when explicitly stated as critical"
    )
    details: Dict[str, str] = Field(
        default_factory=dict,
        description="Structured details about the preference including values, reasons, and context"
    )

class PreferenceExtractorResponse(BaseModel):
    """
    Model for the complete preference extraction response. This model aggregates all preferences
    extracted from a conversation, ensuring they are properly structured and validated.
    
    Attributes:
        preferences (List[PreferenceResponse]): List of all extracted preferences from the conversation.
            Each preference should be atomic (one clear preference per entry) and include all necessary
            context for future reference.
    """
    model_config = {
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "examples": [
                {
                    "preferences": [
                        {
                            "person_name": "Eric Lampron",
                            "preference_type": "location",
                            "importance": 4,
                            "details": {
                                "city": "Montreal",
                                "country": "Canada",
                                "reason": "current residence",
                                "context": "used for weather and local services"
                            }
                        },
                        {
                            "person_name": "Smiles",
                            "preference_type": "communication",
                            "importance": 3,
                            "details": {
                                "style": "casual",
                                "tone": "friendly",
                                "emoji_usage": "moderate",
                                "reason": "preferred interaction style"
                            }
                        }
                    ]
                }
            ]
        }
    }
    preferences: List[PreferenceResponse] = Field(
        ...,
        description="List of extracted preferences, each representing a single, well-defined preference with full context"
    )

class DocumentType(str, Enum):
    """Core document types for SMILE system.
    Note: This is an extensible list - AI can create new types as needed."""
    DOCUMENTATION = "Documentation"  # System or process documentation
    WEB_SUMMARY = "Web Summary"  # Summarized web content
    TASK_GUIDE = "Task Guide"  # How-to guides and instructions
    DATA_FILE = "Data File"  # Structured data files
    SCREENSHOT = "Screenshot"  # Screen captures
    PICTURE = "Picture"  # Images and photos
    DIAGRAM = "Diagram"  # Visual representations and charts
    CONVERSATION = "Conversation"  # Saved chat or discussion
    NOTE = "Note"  # Quick notes or memos
    PLAN = "Plan"  # Project or task plans
    REPORT = "Report"  # Analysis or status reports
    CODE = "Code"  # Code snippets or scripts
    TEMPLATE = "Template"  # Reusable document templates
    OTHER = "Other"  # For custom types not in core list

class SmileDocument(BaseEntity):
    """
    Model representing a document stored in the SMILE library.
    
    Attributes:
        id (UUID): Unique identifier for the document
        name (str): Name of the document (including extension)
        type (str): Type of entity, fixed as 'Document'
        doc_type (str): Type of document (from DocumentType or custom)
        content (str): Content of the document
        file_path (str): Relative path within library folder
        file_url (str): Absolute URL/path to the document file
        file_type (str): File extension/type (e.g., md, txt, png)
        topics (List[str]): List of topics covered in document
        entities (List[str]): Named entities mentioned in document
        created_at (datetime): When the document was created
        updated_at (Optional[datetime]): When the document was last updated
        created_by (Optional[str]): User or agent who created the document
        last_accessed_at (Optional[datetime]): When document was last accessed
        access_count (int): Number of times document was accessed
        version (int): Document version number
        language (str): Document language
        summary (Optional[str]): Brief summary of content
        status (str): Document status (e.g., draft, final)
        tags (List[str]): Custom tags for categorization
        embedding (Optional[List[float]]): Vector embedding for similarity search
        metadata (Dict[str, Any]): Additional flexible metadata
    """
    type: Literal["Document"] = Field(default="Document", description="Type of entity, fixed as 'Document'")
    doc_type: str
    content: str
    file_path: str
    file_url: str
    file_type: str
    topics: List[str] = Field(default=[])
    entities: List[str] = Field(default=[])
    created_by: Optional[str] = None
    last_accessed_at: Optional[datetime] = None
    access_count: int = Field(default=0)
    version: int = Field(default=1)
    language: str = Field(default="en")
    summary: Optional[str] = None
    status: str = Field(default="draft")
    tags: List[str] = Field(default=[])
    metadata: Dict[str, Any] = Field(default={})

    def increment_access(self):
        """Update access metadata when document is accessed."""
        self.access_count += 1
        self.last_accessed_at = datetime.utcnow()

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding."""
        components = [
            self.name,
            self.doc_type,
            self.content,
            self.summary or "",
            " ".join(self.topics),
            " ".join(self.entities),
            " ".join(self.tags),
            " ".join(f"{k}:{v}" for k, v in self.metadata.items())
        ]
        return " ".join(filter(None, components))

    def get_relationships(self) -> List[Relationship]:
        """Get relationships for Neo4j as Relationship objects."""
        relationships = []
        
        # Topic relationships
        for topic in self.topics:
            relationships.append(
                Relationship(
                    from_entity_id=self.id,
                    to_entity_id=uuid4(),  # Temporary ID for the topic
                    type=RelationshipType.COVERS_TOPIC,
                    notes=f"Document covers topic: {topic}"
                )
            )
        
        # Entity relationships
        for entity in self.entities:
            relationships.append(
                Relationship(
                    from_entity_id=self.id,
                    to_entity_id=uuid4(),  # Temporary ID for the entity
                    type=RelationshipType.MENTIONS_ENTITY,
                    notes=f"Document mentions entity: {entity}"
                )
            )
        
        return relationships

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Override model_dump to convert metadata to JSON string."""
        data = super().model_dump(*args, **kwargs)
        if data.get('metadata'):
            data['metadata'] = json.dumps(data['metadata'], cls=Neo4jEncoder)
        return data

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
    type: str = Field(default="general", description="Type of memory (e.g., 'fact', 'preference', 'event')")
    sub_type: Optional[str] = Field(default=None, description="Optional refinement of type")
    
    # Content
    content: str = Field(..., description="Primary textual content of the memory")
    structured_data: Optional[Dict[str, Any]] = Field(default=None, description="Structured representation of the memory content")
    
    # Semantic and Cognitive Information
    semantic: SemanticAttributes = Field(default_factory=SemanticAttributes, description="Semantic attributes of the memory")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding for similarity search")
    
    # Temporal and Source Information
    temporal: TemporalContext = Field(default_factory=TemporalContext, description="Temporal context of the memory")
    source: MemorySource = Field(default=MemorySource.DIRECT_OBSERVATION, description="Source of the memory")
    source_messages: List[str] = Field(default_factory=list, description="Original messages that led to this memory")
    source_id: Optional[str] = Field(default=None, description="ID of the source entity/message")
    
    # Confidence and Validation
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score for the memory")
    validation: ValidationStatus = Field(default_factory=ValidationStatus, description="Validation status of the memory")
    
    # Relations and Associations
    relations: MemoryRelations = Field(default_factory=MemoryRelations, description="Relationships with other system components")
    
    # Context
    context: str = Field(default="general", description="Context in which the memory was formed")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = Field(default=None)
    
    # System Fields
    version: int = Field(default=1)
    is_archived: bool = Field(default=False)
    
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
            f"Context: {self.context}",
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