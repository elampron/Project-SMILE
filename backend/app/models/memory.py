import uuid
from pydantic import BaseModel, Field, field_validator
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
        start_time (datetime): When events begin/began (set programmatically)
        end_time (datetime): When events end/ended (set programmatically)
        embedding (Optional[List[float]]): Vector embedding for search
    """
    id: UUID = Field(default_factory=uuid4)
    content: str
    topics: List[str]
    action_items: List[ActionItem]
    participants: List[Participant]
    sentiments: Dict[str, Any]
    location: Optional[str] = None
    events: List[str] = Field(default_factory=list)
    message_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: datetime = Field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding for similarity search")

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
