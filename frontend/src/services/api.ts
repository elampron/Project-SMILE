import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

export interface Node {
  id: string;
  labels: string[];
  properties: Record<string, any>;
}

export interface Relationship {
  id: string;
  type: string;
  startNode: string;
  endNode: string;
  properties: Record<string, any>;
}

export interface SearchParams {
  query: string;
  labels?: string[];
  limit?: number;
}

export const fetchNodes = async (): Promise<Node[]> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/graph/nodes`);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(`Failed to fetch nodes: ${error.response?.data?.detail || error.message}`);
    }
    throw new Error('Failed to fetch nodes: Unknown error occurred');
  }
};

export const fetchRelationships = async (): Promise<Relationship[]> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/graph/relationships`);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(`Failed to fetch relationships: ${error.response?.data?.detail || error.message}`);
    }
    throw new Error('Failed to fetch relationships: Unknown error occurred');
  }
};

export const searchNodes = async (params: SearchParams): Promise<Node[]> => {
  try {
    const queryParams = new URLSearchParams();
    queryParams.append('query', params.query);
    
    if (params.labels && params.labels.length > 0) {
      params.labels.forEach(label => queryParams.append('labels', label));
    }
    
    if (params.limit) {
      queryParams.append('limit', params.limit.toString());
    }

    const response = await axios.get(`${API_BASE_URL}/api/graph/search?${queryParams.toString()}`);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(`Failed to search nodes: ${error.response?.data?.detail || error.message}`);
    }
    throw new Error('Failed to search nodes: Unknown error occurred');
  }
};