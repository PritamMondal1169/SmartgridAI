import React, { useState } from 'react';

export default function AnomalyMonitor() {
  const [filter, setFilter] = useState('ALL');

  const anomalies = [
    {
      id: 'ANOM-901',
      feeder: 'Feeder C-08',
      type: 'Sudden Voltage Drop (Sag)',
      magnitude: '-18.4% (196V)',
      timestamp: 'Today, 14:22:15',
      confidence: 0.98,
      status: 'Open',
    },
    {
      id: 'ANOM-884',
      feeder: 'Feeder A-01',
      type: 'Reverse Power Flow (Unregistered Solar Export)',
      magnitude: '+14.2 kW',
      timestamp: 'Today, 12:45:00',
      confidence: 0.94,
      status: 'Investigating',
    },
    {
      id: 'ANOM-761',
      feeder: 'Feeder B-12',
      type: 'High-Frequency Harmonic Spike (THD > 8%)',
      magnitude: 'THD 9.2%',
      timestamp: 'Yesterday, 21:05:40',
      confidence: 0.88,
      status: 'Resolved',
    },
  ];

  const filtered = filter === 'ALL' ? anomalies : anomalies.filter((a) => a.status === filter);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Anomaly & Transient Monitor</h1>
          <p className="text-gray-400 text-sm mt-1">Real-time isolation forest & harmonic outlier detection</p>
        </div>

        <div className="flex items-center space-x-2">
          {['ALL', 'Open', 'Investigating', 'Resolved'].map((st) => (
            <button
              key={st}
              onClick={() => setFilter(st)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                filter === st
                  ? 'bg-cyan-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              {st}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-gray-300">
            <thead className="bg-gray-800/80 text-xs uppercase text-gray-400 font-semibold tracking-wider">
              <tr>
                <th className="px-6 py-3.5">Incident ID</th>
                <th className="px-6 py-3.5">Feeder / Meter</th>
                <th className="px-6 py-3.5">Anomaly Classification</th>
                <th className="px-6 py-3.5">Magnitude</th>
                <th className="px-6 py-3.5">Confidence</th>
                <th className="px-6 py-3.5">Timestamp</th>
                <th className="px-6 py-3.5">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {filtered.map((item) => (
                <tr key={item.id} className="hover:bg-gray-800/40 transition">
                  <td className="px-6 py-4 font-mono text-cyan-400 font-semibold text-xs">{item.id}</td>
                  <td className="px-6 py-4 text-white font-medium">{item.feeder}</td>
                  <td className="px-6 py-4">{item.type}</td>
                  <td className="px-6 py-4 font-mono text-rose-400 font-medium">{item.magnitude}</td>
                  <td className="px-6 py-4 font-mono text-xs">{(item.confidence * 100).toFixed(0)}%</td>
                  <td className="px-6 py-4 text-xs text-gray-400">{item.timestamp}</td>
                  <td className="px-6 py-4">
                    <span
                      className={`text-[11px] font-semibold px-2.5 py-1 rounded-full uppercase tracking-wider ${
                        item.status === 'Open'
                          ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                          : item.status === 'Investigating'
                          ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                          : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                      }`}
                    >
                      {item.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

