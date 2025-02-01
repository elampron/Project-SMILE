import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Box,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  useTheme,
} from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import PeopleIcon from '@mui/icons-material/People';
import BusinessIcon from '@mui/icons-material/Business';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import SettingsIcon from '@mui/icons-material/Settings';

interface NavItem {
  path: string;
  icon: React.ReactNode;
  label: string;
}

const Sidebar: React.FC = () => {
  const location = useLocation();
  const theme = useTheme();

  const navItems: NavItem[] = [
    {
      path: '/',
      icon: <DashboardIcon />,
      label: 'Dashboard',
    },
    {
      path: '/people',
      icon: <PeopleIcon />,
      label: 'People',
    },
    {
      path: '/organizations',
      icon: <BusinessIcon />,
      label: 'Organizations',
    },
    {
      path: '/knowledge-explorer',
      icon: <AccountTreeIcon />,
      label: 'Knowledge Explorer',
    },
    {
      path: '/settings',
      icon: <SettingsIcon />,
      label: 'Settings',
    },
  ];

  return (
    <Paper
      sx={{
        width: 240,
        height: '100vh',
        position: 'fixed',
        left: 0,
        top: 0,
        borderRadius: 0,
      }}
    >
      <Box sx={{ p: 2 }}>
        <img
          src="/logo.png"
          alt="Project SMILE Logo"
          style={{ width: '100%', height: 'auto' }}
        />
      </Box>
      <List>
        {navItems.map((item) => (
          <ListItem
            key={item.path}
            component={Link}
            to={item.path}
            sx={{
              color: location.pathname === item.path ? theme.palette.primary.main : 'inherit',
              bgcolor: location.pathname === item.path ? theme.palette.action.selected : 'transparent',
              '&:hover': {
                bgcolor: theme.palette.action.hover,
              },
            }}
          >
            <ListItemIcon
              sx={{
                color: location.pathname === item.path ? theme.palette.primary.main : 'inherit',
              }}
            >
              {item.icon}
            </ListItemIcon>
            <ListItemText primary={item.label} />
          </ListItem>
        ))}
      </List>
    </Paper>
  );
};

export default Sidebar;