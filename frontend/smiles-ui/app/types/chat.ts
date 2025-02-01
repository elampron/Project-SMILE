/**
 * Represents a chat message in the application
 */
export interface Message {
  /** The content of the message */
  content: string;
  /** The role of the message sender - either 'user' or 'assistant' */
  role: 'user' | 'assistant';
  /** The timestamp when the message was sent */
  timestamp: Date;
  /** Optional thread ID for message grouping */
  thread_id?: string;
  /** Optional array of file names attached to the message */
  files?: string[];
} 