import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';

export default function NavBar({ dark, setDark }) {
  const navigate = useNavigate();
  const username = localStorage.getItem('username') || 'Operator';
  const role = localStorage.getItem('role') || 'Admin';

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('role');
    localStorage.removeItem('username');
    navigate('/login');
  };

  const navItems = [
    { name: 'Overview', path: '/overview' },
    { name: 'Analytics', path: '/analytics' },
    { name: 'Consumption', path: '/consumption' },
    { name: 'Model Lab', path: '/models' },
    { name: 'Inspection Queue', path: '/inspection' },
    { name: 'Anomalies', path: '/anomalies' },
  ];

  return (
    <nav className="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-3 cursor-pointer" onClick={() => navigate('/overview')}>
              <div className="w-8 h-8 rounded-lg bg-cyan-500/20 text-cyan-400 flex items-center justify-center font-bold border border-cyan-500/30">
                ⚡
              </div>
              <span className="font-bold text-white text-base tracking-tight">SmartgridAI</span>
            </div>

            <div className="hidden md:flex space-x-1">
              {navItems.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    `px-3 py-2 rounded-lg text-xs font-medium transition ${
                      isActive
                        ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
                        : 'text-gray-400 hover:text-white hover:bg-gray-800/60'
                    }`
                  }
                >
                  {item.name}
                </NavLink>
              ))}
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="hidden sm:flex flex-col text-right">
              <span className="text-xs font-semibold text-white">{username}</span>
              <span className="text-[10px] text-cyan-400 uppercase tracking-wider">{role}</span>
            </div>

            <button
              onClick={() => setDark(!dark)}
              className="p-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-300 hover:text-white text-xs transition"
              title="Toggle Dark / Light Mode"
            >
              {dark ? '🌙' : '☀️'}
            </button>

            <button
              onClick={logout}
              className="px-3 py-1.5 rounded-lg bg-rose-600/10 border border-rose-500/20 text-rose-400 hover:bg-rose-600 hover:text-white text-xs font-medium transition"
            >
              Sign Out
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
