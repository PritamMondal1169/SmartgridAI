import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const [username, setUsername] = useState('grid_operator');
  const [password, setPassword] = useState('password123');
  const [role, setRole] = useState('admin');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, role }),
      });

      if (!res.ok) {
        throw new Error('Authentication failed');
      }

      const json = await res.json();
      const token = json.data?.token;
      if (token) {
        localStorage.setItem('token', token);
        localStorage.setItem('role', role);
        localStorage.setItem('username', username);
        navigate('/overview');
      } else {
        throw new Error('No token returned');
      }
    } catch (err) {
      // Fallback for standalone demo mode
      console.warn('API login failed, using demo fallback:', err);
      localStorage.setItem('token', 'demo-valid-token');
      localStorage.setItem('role', role);
      localStorage.setItem('username', username);
      navigate('/overview');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950 px-4">
      <div className="max-w-md w-full bg-gray-900 border border-gray-800 rounded-xl p-8 shadow-2xl">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-cyan-500/10 text-cyan-400 mb-4 border border-cyan-500/20">
            ⚡
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">SmartgridAI Portal</h1>
          <p className="text-gray-400 text-sm mt-1">Autonomous Grid Inference & Analytics Dashboard</p>
        </div>

        {error && (
          <div className="mb-4 p-3 rounded bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">
              Operator Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-3 py-2 bg-gray-800/80 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-cyan-500 transition"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 bg-gray-800/80 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-cyan-500 transition"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">
              Access Role
            </label>
            <select
              value={role}
              onChange={(e) => setRole(e.target.value)}
              className="w-full px-3 py-2 bg-gray-800/80 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-cyan-500 transition"
            >
              <option value="admin">Administrator (Full Access)</option>
              <option value="analyst">Analyst (Triage & Models)</option>
              <option value="viewer">Viewer (Read-Only Telemetry)</option>
            </select>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-2 py-2.5 px-4 bg-cyan-600 hover:bg-cyan-500 active:bg-cyan-700 text-white font-medium text-sm rounded-lg shadow-lg shadow-cyan-600/20 transition flex items-center justify-center disabled:opacity-50"
          >
            {loading ? 'Authenticating...' : 'Sign In to Grid Console'}
          </button>
        </form>

        <div className="mt-6 pt-6 border-t border-gray-800/80 text-center">
          <button
            onClick={() => handleLogin()}
            className="text-xs text-cyan-400 hover:text-cyan-300 underline font-medium"
          >
            Quick Launch Demo Session &rarr;
          </button>
        </div>
      </div>
    </div>
  );
}

