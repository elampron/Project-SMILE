'use client';

import { useState } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { 
    ChatBubbleLeftIcon, 
    MicrophoneIcon, 
    BookOpenIcon, 
    Cog6ToothIcon,
    ChevronLeftIcon,
    ChevronRightIcon 
} from '@heroicons/react/24/outline';

interface NavItem {
    name: string;
    path: string;
    icon: typeof ChatBubbleLeftIcon;
}

const navItems: NavItem[] = [
    { name: 'Chat', path: '/', icon: ChatBubbleLeftIcon },
    { name: 'Voice', path: '/voice', icon: MicrophoneIcon },
    { name: 'Knowledge', path: '/knowledge', icon: BookOpenIcon },
    { name: 'Admin', path: '/admin', icon: Cog6ToothIcon },
];

export default function Sidebar() {
    const [isCollapsed, setIsCollapsed] = useState(false);
    const router = useRouter();
    const pathname = usePathname();

    return (
        <div 
            className={`h-screen bg-black border-r border-green-500/20 text-green-500 transition-all duration-300 flex flex-col ${
                isCollapsed ? 'w-16' : 'w-64'
            }`}
        >
            {/* Toggle button */}
            <button
                onClick={() => setIsCollapsed(!isCollapsed)}
                className="p-4 hover:bg-green-500/10 self-end transition-colors"
            >
                {isCollapsed ? (
                    <ChevronRightIcon className="h-6 w-6" />
                ) : (
                    <ChevronLeftIcon className="h-6 w-6" />
                )}
            </button>

            {/* Navigation items */}
            <nav className="flex-1">
                {navItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = pathname === item.path;
                    
                    return (
                        <button
                            key={item.name}
                            onClick={() => router.push(item.path)}
                            className={`w-full flex items-center p-4 transition-colors
                                ${isActive 
                                    ? 'bg-green-500/20 text-green-400' 
                                    : 'hover:bg-green-500/10'
                                }`}
                        >
                            <Icon className="h-6 w-6 flex-shrink-0" />
                            {!isCollapsed && (
                                <span className="ml-4 font-mono">{item.name}</span>
                            )}
                        </button>
                    );
                })}
            </nav>
        </div>
    );
} 