import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

import NavBar from './components/NavBar';
import ProtectedRoute from './components/ProtectedRoute';
import Login from './pages/Login';
import Overview from './pages/Overview';
import Analytics from './pages/Analytics';
import ConsumptionChart from './pages/ConsumptionChart';
import ModelLab from './pages/ModelLab';
import InspectionQueue from './pages/InspectionQueue';
import AnomalyMonitor from './pages/AnomalyMonitor';

export default function App() {
  const [dark, setDark] = useState(true);

  useEffect(() => {
    if (dark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [dark]);

  return (
    <BrowserRouter>
      <div className={`min-h-screen ${dark ? 'dark bg-gray-950 text-white' : 'bg-gray-50 text-gray-900'}`}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <div className="flex flex-col min-h-screen">
                  <NavBar dark={dark} setDark={setDark} />
                  <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
                    <Routes>
                      <Route path="/overview" element={<Overview />} />
                      <Route path="/analytics" element={<Analytics />} />
                      <Route path="/consumption" element={<ConsumptionChart />} />
                      <Route path="/models" element={<ModelLab />} />
                      <Route path="/inspection" element={<InspectionQueue />} />
                      <Route path="/anomalies" element={<AnomalyMonitor />} />
                      <Route path="*" element={<Navigate to="/overview" replace />} />
                    </Routes>
                  </main>
                  <footer className="border-t border-gray-800/80 py-4 text-center text-xs text-gray-500">
                    SmartgridAI Grid Telemetry & Inference Platform &copy; 2026
                  </footer>
                </div>
              </ProtectedRoute>
            }
          />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

