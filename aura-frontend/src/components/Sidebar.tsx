/**
 * Sidebar navigation component
 */
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Upload,
  FolderSearch,
  Activity,
  ShieldAlert,
  Database,
  BarChart3,
  Shield,
  X,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import type { SidebarProps } from '../types';
import { cn } from '../utils/cn';

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Dataset Explorer', href: '/dataset-explorer', icon: FolderSearch },
  { name: 'Submit Model', href: '/upload', icon: Upload },
  { name: 'Sentinel Monitor', href: '/sentinel-monitor', icon: Activity },
  { name: 'Attack Evidence', href: '/attack-evidence', icon: ShieldAlert },
  { name: 'Blockchain Ledger', href: '/ledger', icon: Database },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'Security Center', href: '/security', icon: Shield },
];

export function Sidebar({ isOpen, onToggle }: SidebarProps) {
  return (
    <>
      {/* Mobile overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onToggle}
            className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <aside
        className={cn(
          'fixed top-0 left-0 z-50 h-screen w-64',
          'bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800',
          'flex flex-col',
          // Mobile: hidden by default, show when open
          'transition-transform duration-300 ease-in-out',
          isOpen ? 'translate-x-0' : '-translate-x-full',
          // Desktop: always visible
          'lg:translate-x-0'
        )}
      >
        {/* Logo */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">AURA</h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">Security Dashboard</p>
            </div>
          </div>
          
          <button
            onClick={onToggle}
            className="lg:hidden text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) =>
                cn(
                  'relative flex items-center gap-3 px-4 py-3 rounded-lg transition-all',
                  'text-gray-700 dark:text-gray-300',
                  'hover:bg-gray-100 dark:hover:bg-gray-800',
                  isActive &&
                    'bg-primary-50 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400 font-medium'
                )
              }
            >
              {({ isActive }) => (
                <>
                  <item.icon className="w-5 h-5 flex-shrink-0" />
                  <span>{item.name}</span>
                  {isActive && (
                    <div className="absolute right-0 w-1 h-8 bg-primary-600 rounded-l-full" />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* User profile */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-800">
          <div className="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
            <div className="w-10 h-10 rounded-full bg-primary-600 flex items-center justify-center text-white font-semibold">
              HA
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                Hospital Admin
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                admin@hospital.com
              </p>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
