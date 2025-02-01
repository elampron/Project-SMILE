import { NextRequest, NextResponse } from 'next/server';

/**
 * Fetches graph data from the backend based on search parameters
 * @param req The Next.js request object
 * @returns A JSON response containing nodes and links for the graph visualization
 */
export async function GET(req: NextRequest) {
  try {
    const searchParams = req.nextUrl.searchParams;
    const search = searchParams.get('search') || '';
    const nodeType = searchParams.get('nodeType');

    // Construct the backend API URL with query parameters
    const queryParams = new URLSearchParams();
    if (search) queryParams.append('search', search);
    if (nodeType) queryParams.append('nodeType', nodeType);

    // Get the backend URL from environment variables
    const backendUrl = process.env.SMILES_API_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
    
    // Fetch data from the backend using the correct endpoint path
    const response = await fetch(`${backendUrl}/api/v1/graph/explore?${queryParams.toString()}`, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    
    // Transform the data to match the frontend's expected format
    const transformedData = {
      nodes: data.nodes.map((node: any) => ({
        id: node.id,
        label: node.properties.name || node.id,
        type: node.labels[0].toLowerCase(),
        properties: node.properties,
      })),
      links: data.relationships.map((rel: any) => ({
        source: rel.startNode,
        target: rel.endNode,
        type: rel.type.toLowerCase(),
      })),
    };

    return NextResponse.json(transformedData);
  } catch (error) {
    console.error('Error fetching graph data:', error);
    return NextResponse.json(
      { error: 'Failed to fetch graph data' },
      { status: 500 }
    );
  }
} 
