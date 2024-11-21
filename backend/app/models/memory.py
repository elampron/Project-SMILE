import uuid
from pydantic import BaseModel, Field
from typing import List, Optional, Literal,Dict
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum

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
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the entity.")
    db_id: Optional[str] = Field(None, description="Database ID of the entity.")
    name: str = Field(..., description="Name of the entity.")
    type: str = Field(..., description="Type of the entity (e.g., 'Person', 'Organization').")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the entity was created.")
    updated_at: Optional[datetime] = Field(None, description="Timestamp when the entity was last updated.")

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
    Model representing an action item extracted from the conversation.

    Attributes:
        description (str): Description of the action item.
        assignee (Optional[str]): Person responsible for the action.
        due_date (Optional[datetime]): Deadline for the action item.
    """
    description: str
    assignee: Optional[str]
    due_date: Optional[datetime]

class Participant(BaseModel):
    """
    Model representing a participant in the conversation.

    Attributes:
        name (str): Name of the participant.
        role (Optional[str]): Role or relationship to the user (e.g., friend, colleague).
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

class Timeframe(BaseModel):
    """
    Model representing time-related information from the conversation.

    Attributes:
        start_date (Optional[datetime]): Start date mentioned.
        end_date (Optional[datetime]): End date mentioned.
    """
    start_date: Optional[datetime]
    end_date: Optional[datetime]

class ConversationSummary(BaseModel):
    """
    Model representing a summary of a conversation batch.

    Attributes:
        id (UUID): Unique identifier for the summary.
        content (str): The generated summary text.
        topics (List[str]): Main topics discussed.
        action_items (List[ActionItem]): List of action items identified.
        participants (List[Participant]): Participants involved in the conversation.
        sentiments (Optional[SentimentAnalysis]): Sentiment analysis of the conversation.
        timeframe (Optional[Timeframe]): Time-related information mentioned.
        location (Optional[str]): Locations mentioned in the conversation.
        events (List[str]): Significant events referenced.
        created_at (datetime): Timestamp when the summary was created.
        tool_interactions (List[str]): Short descriptions of tool interactions included in the summary.
    """
    id: UUID = Field(default_factory=uuid4)
    content: str
    topics: List[str]
    action_items: List['ActionItem']
    participants: List['Participant']
    sentiments: Optional['SentimentAnalysis']
    timeframe: Optional['Timeframe']
    location: Optional[str]
    events: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tool_interactions: List[str] = [] 

class Preference(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    id: uuid.UUID = uuid.uuid4()
    person_id: uuid.UUID
    preference_type: str
    importance: int
    details: Dict[str, any]
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

class PreferenceResponse(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True  # Add this configuration
    }
    person_name: str
    preference_type: str
    importance: int
    details: Dict[str, any]

class PreferenceExtractorResponse(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True  # Add this configuration
    }
    preferences: List[PreferenceResponse]
