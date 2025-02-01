import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { Box, TextField, FormControl, InputLabel, Select, MenuItem, Paper, Typography, CircularProgress } from '@mui/material';
import ForceGraph2D from 'react-force-graph-2d';
import { useTheme } from '@mui/material/styles';
import useMediaQuery from '@mui/material/useMediaQuery';
import { debounce } from 'lodash';
import axios from 'axios';

interface Node {
  id: string;
  label: string;
  type: string;
  properties: Record<string, any>;
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

const KnowledgeExplorer: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedNodeType, setSelectedNodeType] = useState('all');
  const [error, setError] = useState<string | null>(null);

  const nodeTypes = useMemo(() => {
    const types = new Set(graphData.nodes.map(node => node.type));
    return ['all', ...Array.from(types)];
  }, [graphData.nodes]);

  const fetchGraphData = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/graph/explore', {
        params: {
          search: searchTerm,
          nodeType: selectedNodeType !== 'all' ? selectedNodeType : undefined
        }
      });
      setGraphData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to load graph data. Please try again later.');
      console.error('Error fetching graph data:', err);
    } finally {
      setLoading(false);
    }
  }, [searchTerm, selectedNodeType]);

  const debouncedFetch = useMemo(
    () => debounce(fetchGraphData, 500),
    [fetchGraphData]
  );

  useEffect(() => {
    debouncedFetch();
    return () => debouncedFetch.cancel();
  }, [debouncedFetch]);

  const handleNodeClick = useCallback((node: Node) => {
    // Handle node click - could show details in a modal or sidebar
    console.log('Clicked node:', node);
  }, []);

  const filteredGraphData = useMemo(() => {
    if (selectedNodeType === 'all') return graphData;
    
    const filteredNodes = graphData.nodes.filter(node => node.type === selectedNodeType);
    const nodeIds = new Set(filteredNodes.map(node => node.id));
    
    const filteredLinks = graphData.links.filter(
      link => nodeIds.has(link.source as string) && nodeIds.has(link.target as string)
    );

    return { nodes: filteredNodes, links: filteredLinks };
  }, [graphData, selectedNodeType]);

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', p: 2 }}>
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h5" sx={{ mb: 2 }}>
          Knowledge Explorer
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <TextField
            label="Search nodes"
            variant="outlined"
            size="small"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            sx={{ flexGrow: 1 }}
          />
          
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Node Type</InputLabel>
            <Select
              value={selectedNodeType}
              label="Node Type"
              onChange={(e) => setSelectedNodeType(e.target.value)}
            >
              {nodeTypes.map((type) => (
                <MenuItem key={type} value={type}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      </Paper>

      {error && (
        <Paper sx={{ p: 2, mb: 2, bgcolor: 'error.light' }}>
          <Typography color="error">{error}</Typography>
        </Paper>
      )}

      <Paper sx={{ flexGrow: 1, position: 'relative', overflow: 'hidden' }}>
        {loading ? (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)'
            }}
          >
            <CircularProgress />
          </Box>
        ) : (
          <ForceGraph2D
            graphData={filteredGraphData}
            nodeLabel={(node: Node) => `${node.label} (${node.type})`}
            nodeColor={(node: Node) => {
              switch (node.type) {
                case 'person':
                  return theme.palette.primary.main;
                case 'organization':
                  return theme.palette.secondary.main;
                default:
                  return theme.palette.grey[500];
              }
            }}
            width={isMobile ? window.innerWidth - 32 : window.innerWidth - 48}
            height={window.innerHeight - 200}
            onNodeClick={handleNodeClick}
            linkColor={() => theme.palette.grey[400]}
            nodeCanvasObject={(node: Node, ctx, globalScale) => {
              const label = node.label;
              const fontSize = 12/globalScale;
              ctx.font = `${fontSize}px Sans-Serif`;
              ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
              ctx.fillText(label, node.x! + 8/globalScale, node.y!);
            }}
          />
        )}
      </Paper>
    </Box>
  );
};

export default KnowledgeExplorer;