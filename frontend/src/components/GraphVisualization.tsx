import React, { useCallback, useRef, useState } from 'react';
import ForceGraph2D, { ForceGraphMethods } from 'react-force-graph-2d';
import { NodeObject, LinkObject } from 'force-graph';

interface GraphNode extends NodeObject {
  id: string;
  label: string;
  type: string;
  properties?: Record<string, any>;
}

interface GraphLink extends LinkObject {
  source: string;
  target: string;
  type: string;
  properties?: Record<string, any>;
}

interface GraphVisualizationProps {
  nodes: GraphNode[];
  links: GraphLink[];
  onNodeClick?: (node: GraphNode) => void;
  onLinkClick?: (link: GraphLink) => void;
  width?: number;
  height?: number;
}

const GraphVisualization: React.FC<GraphVisualizationProps> = ({
  nodes,
  links,
  onNodeClick,
  onLinkClick,
  width = 800,
  height = 600,
}) => {
  const graphRef = useRef<ForceGraphMethods>();
  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  const updateHighlight = useCallback(() => {
    setHighlightNodes(highlightNodes);
    setHighlightLinks(highlightLinks);
  }, [highlightNodes, highlightLinks]);

  const handleNodeHover = useCallback((node: GraphNode | null) => {
    highlightNodes.clear();
    highlightLinks.clear();
    if (node) {
      highlightNodes.add(node);
      links
        .filter((link) => link.source === node.id || link.target === node.id)
        .forEach((link) => {
          highlightLinks.add(link);
          highlightNodes.add(nodes.find((n) => n.id === link.source) || null);
          highlightNodes.add(nodes.find((n) => n.id === link.target) || null);
        });
    }
    updateHighlight();
  }, [links, nodes, highlightLinks, highlightNodes, updateHighlight]);

  const handleNodeClick = useCallback((node: GraphNode) => {
    if (node) {
      const distance = 40;
      const distRatio = 1 + distance / Math.hypot(node.x || 0, node.y || 0);
      graphRef.current?.centerAt(node.x, node.y, 1000);
      graphRef.current?.zoom(2.5, 1000);
      setSelectedNode(node);
      if (onNodeClick) onNodeClick(node);
    }
  }, [onNodeClick]);

  const handleLinkClick = useCallback((link: GraphLink) => {
    if (onLinkClick) onLinkClick(link);
  }, [onLinkClick]);

  return (
    <div style={{ border: '1px solid #ddd', borderRadius: '4px' }}>
      <ForceGraph2D
        ref={graphRef}
        graphData={{ nodes, links }}
        nodeLabel={(node: GraphNode) => `${node.label} (${node.type})`}
        nodeColor={(node: GraphNode) =>
          highlightNodes.has(node) ? '#f50057' : '#1976d2'
        }
        nodeRelSize={6}
        linkWidth={(link) => (highlightLinks.has(link) ? 2 : 1)}
        linkColor={(link) => (highlightLinks.has(link) ? '#f50057' : '#999')}
        linkLabel={(link: GraphLink) => link.type}
        onNodeHover={handleNodeHover}
        onNodeClick={handleNodeClick}
        onLinkClick={handleLinkClick}
        width={width}
        height={height}
        enableNodeDrag={true}
        enableZoomPanInteraction={true}
        cooldownTicks={100}
        onEngineStop={() => {
          graphRef.current?.zoomToFit(400, 50);
        }}
        linkDirectionalParticles={2}
        linkDirectionalParticleWidth={2}
        d3VelocityDecay={0.3}
      />
    </div>
  );
};

export default GraphVisualization;