// Mark this component as a server component
export const dynamic = 'force-dynamic'; // Opt out of static rendering
export const fetchCache = 'force-no-store'; // Disable caching

import { Suspense } from 'react';
import { KnowledgeExplorer } from '@/app/knowledge/KnowledgeExplorer';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Knowledge Explorer',
  description: 'Explore and visualize knowledge graph data',
};

interface SearchParams {
  search?: string;
  nodeType?: string;
}

async function getGraphData(searchParams: SearchParams) {
  const backendUrl = process.env.SMILES_API_URL || 'http://backend:8000';
  
  const queryParams = new URLSearchParams();
  if (searchParams.search) queryParams.append('search', searchParams.search);
  if (searchParams.nodeType && searchParams.nodeType !== 'all') {
    queryParams.append('nodeType', searchParams.nodeType);
  }

  try {
    const response = await fetch(
      `${backendUrl}/api/v1/graph/explore?${queryParams.toString()}`,
      {
        headers: {
          'Content-Type': 'application/json',
        },
        cache: 'no-store',
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch graph data: ${response.status}`);
    }

    const data = await response.json();
    return {
      nodes: data.nodes?.map((node: any) => ({
        id: node.id,
        label: node.properties.name || node.id,
        type: node.labels[0].toLowerCase(),
        properties: node.properties,
      })) || [],
      links: data.relationships?.map((rel: any) => ({
        source: rel.startNode,
        target: rel.endNode,
        type: rel.type.toLowerCase(),
      })) || [],
    };
  } catch (error) {
    console.error('Error fetching graph data:', error);
    return { nodes: [], links: [] };
  }
}

export default async function Page({ searchParams }: { searchParams: any }) {
  const search = typeof searchParams?.search === 'string' ? searchParams.search : undefined;
  const nodeType = typeof searchParams?.nodeType === 'string' ? searchParams.nodeType : undefined;
  
  // Await the graph data before rendering
  const graphData = await getGraphData({ search, nodeType });

  return (
    <div className="flex-1 p-8 space-y-4">
      <h1 className="text-2xl font-bold">Knowledge Explorer</h1>
      <Suspense fallback={<div>Loading graph...</div>}>
        <KnowledgeExplorer initialData={graphData} />
      </Suspense>
    </div>
  );
} 