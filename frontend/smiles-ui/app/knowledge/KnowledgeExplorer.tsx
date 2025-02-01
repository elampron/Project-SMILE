'use client';

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { debounce } from 'lodash';
import { Loader2 } from 'lucide-react';

// Dynamically import ForceGraph2D to avoid SSR issues
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-[600px]">
      <Loader2 className="h-8 w-8 animate-spin" />
    </div>
  ),
}) as any; // Type assertion needed due to dynamic import

interface Node {
  id: string;
  label: string;
  type: string;
  properties: Record<string, any>;
  x?: number;
  y?: number;
}

interface Link {
  source: string;
  target: string;
  type: string;
}

interface GraphData {
  nodes: Node[];
  links: Link[];
}

interface KnowledgeExplorerProps {
  initialData: GraphData;
}

const defaultGraphData: GraphData = {
  nodes: [],
  links: []
};

export function KnowledgeExplorer({ initialData = defaultGraphData }: KnowledgeExplorerProps) {
  const [graphData, setGraphData] = useState<GraphData>(initialData);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedNodeType, setSelectedNodeType] = useState('all');
  const [error, setError] = useState<string | null>(null);

  // Update graph data when initialData changes
  useEffect(() => {
    setGraphData(initialData);
  }, [initialData]);

  const nodeTypes = useMemo(() => {
    if (!graphData?.nodes) return ['all'];
    const types = new Set(graphData.nodes.map(node => node.type));
    return ['all', ...Array.from(types)];
  }, [graphData?.nodes]);

  const updateSearchParams = useCallback(async () => {
    try {
      // Update URL with search parameters without page reload
      const url = new URL(window.location.href);
      url.searchParams.set('search', searchTerm);
      if (selectedNodeType !== 'all') {
        url.searchParams.set('nodeType', selectedNodeType);
      } else {
        url.searchParams.delete('nodeType');
      }
      window.history.pushState({}, '', url.toString());
      
      // Server will rerender with new data due to searchParams change
    } catch (err) {
      setError('Failed to update search parameters. Please try again later.');
      console.error('Error updating search:', err);
    }
  }, [searchTerm, selectedNodeType]);

  const debouncedUpdate = useMemo(
    () => debounce(updateSearchParams, 500),
    [updateSearchParams]
  );

  const handleNodeClick = useCallback((node: Node) => {
    // Handle node click - could show details in a modal or sidebar
    console.log('Clicked node:', node);
  }, []);

  const filteredGraphData = useMemo(() => {
    if (!graphData?.nodes || !graphData?.links) return defaultGraphData;
    if (selectedNodeType === 'all') return graphData;
    
    const filteredNodes = graphData.nodes.filter(node => node.type === selectedNodeType);
    const nodeIds = new Set(filteredNodes.map(node => node.id));
    
    const filteredLinks = graphData.links.filter(
      link => nodeIds.has(link.source as string) && nodeIds.has(link.target as string)
    );

    return { nodes: filteredNodes, links: filteredLinks };
  }, [graphData, selectedNodeType]);

  if (!graphData?.nodes) {
    return (
      <div className="flex items-center justify-center h-[600px]">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold">Search and Filter</h2>
        </div>
        <div className="p-4 space-y-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <input
              type="text"
              placeholder="Search nodes..."
              value={searchTerm}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                setSearchTerm(e.target.value);
                debouncedUpdate();
              }}
              className="flex-grow px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
            />
            
            <select
              value={selectedNodeType}
              onChange={(e) => {
                setSelectedNodeType(e.target.value);
                debouncedUpdate();
              }}
              className="w-[180px] px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
            >
              {nodeTypes.map((type) => (
                <option key={type} value={type}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 p-4 rounded">
          <p className="text-red-700 dark:text-red-400">{error}</p>
        </div>
      )}

      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 relative min-h-[600px]">
        <div className="p-0">
          <ForceGraph2D
            graphData={filteredGraphData}
            nodeLabel={(node: Node) => `${node.label} (${node.type})`}
            nodeColor={(node: Node) => {
              switch (node.type) {
                case 'person':
                  return '#0ea5e9'; // sky-500
                case 'organization':
                  return '#8b5cf6'; // violet-500
                default:
                  return '#6b7280'; // gray-500
              }
            }}
            width={typeof window !== 'undefined' ? window.innerWidth - 100 : 800}
            height={600}
            onNodeClick={handleNodeClick}
            linkColor={() => '#9ca3af'} // gray-400
            nodeCanvasObject={(node: Node, ctx: CanvasRenderingContext2D, globalScale: number) => {
              const label = node.label;
              const fontSize = 12/globalScale;
              ctx.font = `${fontSize}px Inter`;
              ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
              ctx.fillText(label, node.x! + 8/globalScale, node.y!);
            }}
          />
        </div>
      </div>
    </div>
  );
} 