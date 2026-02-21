/**
 * Main layout component with sidebar and navbar
 */
import { useEffect, useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Navbar } from './Navbar';
import { systemAPI } from '../services/api';

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [systemMode, setSystemMode] = useState<'real' | 'unknown'>('unknown');

  useEffect(() => {
    let cancelled = false;
    const loadMode = async () => {
      try {
        const modeResp = await systemAPI.getMode();
        if (!cancelled) {
          const mode = modeResp.mode === 'real' ? 'real' : 'unknown';
          setSystemMode(mode);
        }
      } catch {
        if (!cancelled) {
          setSystemMode('unknown');
        }
      }
    };
    loadMode();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="flex min-h-screen bg-gray-50 dark:bg-gray-900">
      <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />
      
      <div className="flex-1 flex flex-col lg:pl-64">
        <Navbar onMenuClick={() => setSidebarOpen(!sidebarOpen)} systemMode={systemMode} />
        
        <main className="flex-1 p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
