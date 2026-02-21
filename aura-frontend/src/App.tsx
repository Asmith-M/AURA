/**
 * Main App component with routing
 */
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Upload } from './pages/Upload';
import { Reports } from './pages/Reports';
import { Ledger } from './pages/Ledger';
import { Analytics } from './pages/Analytics';
import { Security } from './pages/Security';
import { SentinelMonitor } from './pages/SentinelMonitor';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="upload" element={<Upload />} />
          <Route path="reports" element={<Reports />} />
          <Route path="ledger" element={<Ledger />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="security" element={<Security />} />
          <Route path="sentinel-monitor" element={<SentinelMonitor />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
