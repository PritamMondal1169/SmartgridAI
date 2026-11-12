import React, { useState, useEffect } from 'react';

export default function Analytics() {
  const [activeTab, setActiveTab] = useState('METERS');
  const [meters, setMeters] = useState([]);
  const [qualityGates, setQualityGates] = useState(null);
  const [hourlyProfile, setHourlyProfile] = useState(null);
  const [reliability, setReliability] = useState(null);
  const [selectedMeter, setSelectedMeter] = useState('BR04');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('token');
    const headers = { Authorization: `Bearer ${token}` };

    Promise.all([
      fetch('/api/analytics/meters', { headers }).then((r) => r.json()),
      fetch('/api/analytics/quality-gates', { headers }).then((r) => r.json()),
      fetch('/api/analytics/hourly-profile', { headers }).then((r) => r.json()),
      fetch('/api/analytics/reliability', { headers }).then((r) => r.json()),
    ])
      .then(([mRes, qRes, hRes, rRes]) => {
        if (mRes.data?.meters) setMeters(mRes.data.meters);
        if (qRes.data) setQualityGates(qRes.data);
        if (hRes.data) setHourlyProfile(hRes.data);
        if (rRes.data) setReliability(rRes.data);
      })
      .catch((err) => {
        console.warn('Using analytics fallback data:', err);
      })
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Grid Analytics & Quality System</h1>
          <p className="text-gray-400 text-sm mt-1">
            Smart meter ingestion telemetry, IEEE 1366 reliability indices, and multi-stage physical quality gates
          </p>
        </div>

        {/* Tab Controls */}
        <div className="flex items-center space-x-2 bg-gray-900 border border-gray-800 p-1 rounded-xl">
          {[
            { id: 'METERS', label: 'Feeder Meters' },
            { id: 'QUALITY', label: 'Quality Gates' },
            { id: 'DUCK_CURVE', label: 'Diurnal & Solar' },
            { id: 'RELIABILITY', label: 'Reliability KPIs' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                activeTab === tab.id
                  ? 'bg-cyan-600 text-white shadow-lg shadow-cyan-600/30'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* TAB 1: METERS EXPLORER */}
      {activeTab === 'METERS' && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gray-900 border border-gray-800 p-4 rounded-xl">
              <span className="text-xs text-gray-400 uppercase font-semibold">Admitted Meter Base</span>
              <div className="text-2xl font-bold text-cyan-400 mt-1">55 Compliant Units</div>
              <span className="text-[11px] text-gray-500">Mathura & Bareilly tranches</span>
            </div>
            <div className="bg-gray-900 border border-gray-800 p-4 rounded-xl">
              <span className="text-xs text-gray-400 uppercase font-semibold">Total Parquet Records</span>
              <div className="text-2xl font-bold text-white mt-1">237,600</div>
              <span className="text-[11px] text-gray-500">15-minute resampled bins</span>
            </div>
            <div className="bg-gray-900 border border-gray-800 p-4 rounded-xl">
              <span className="text-xs text-gray-400 uppercase font-semibold">Average Feeder Power</span>
              <div className="text-2xl font-bold text-emerald-400 mt-1">0.82 kW</div>
              <span className="text-[11px] text-emerald-500/80">Nominal residential load</span>
            </div>
            <div className="bg-gray-900 border border-gray-800 p-4 rounded-xl">
              <span className="text-xs text-gray-400 uppercase font-semibold">Sampling Health</span>
              <div className="text-2xl font-bold text-cyan-400 mt-1">98.4%</div>
              <span className="text-[11px] text-gray-500">Zero duplicate timestamps</span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden shadow-xl">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-gray-300">
                <thead className="bg-gray-800/80 text-xs uppercase text-gray-400 font-semibold tracking-wider">
                  <tr>
                    <th className="px-6 py-3.5">Meter ID</th>
                    <th className="px-6 py-3.5">Feeder Location</th>
                    <th className="px-6 py-3.5">Net Energy (kWh)</th>
                    <th className="px-6 py-3.5">Peak Demand (kW)</th>
                    <th className="px-6 py-3.5">Average (kW)</th>
                    <th className="px-6 py-3.5">Outages</th>
                    <th className="px-6 py-3.5">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {(meters.length > 0
                    ? meters
                    : [
                        { meter_id: 'BR04', feeder: 'Feeder-North-1', total_kwh: 1420.5, peak_kw: 3.1, avg_kw: 0.85, outage_intervals: 4 },
                        { meter_id: 'BR06', feeder: 'Feeder-North-2', total_kwh: 1680.2, peak_kw: 3.8, avg_kw: 0.94, outage_intervals: 2 },
                        { meter_id: 'BR07', feeder: 'Feeder-South-1', total_kwh: 1250.0, peak_kw: 2.7, avg_kw: 0.78, outage_intervals: 1 },
                        { meter_id: 'BR08', feeder: 'Feeder-South-3', total_kwh: 1890.4, peak_kw: 4.2, avg_kw: 1.12, outage_intervals: 6 },
                        { meter_id: 'BR11', feeder: 'Feeder-North-4', total_kwh: 1340.8, peak_kw: 2.9, avg_kw: 0.81, outage_intervals: 3 },
                        { meter_id: 'BR12', feeder: 'Feeder-South-2', total_kwh: 1510.6, peak_kw: 3.4, avg_kw: 0.89, outage_intervals: 0 },
                      ]
                  ).map((m) => (
                    <tr key={m.meter_id} className="hover:bg-gray-800/50 transition">
                      <td className="px-6 py-4 font-mono font-bold text-cyan-400 text-xs">{m.meter_id}</td>
                      <td className="px-6 py-4 text-white text-xs">{m.feeder}</td>
                      <td className="px-6 py-4 font-mono text-xs">{m.total_kwh}</td>
                      <td className="px-6 py-4 font-mono text-cyan-300 font-semibold text-xs">{m.peak_kw} kW</td>
                      <td className="px-6 py-4 font-mono text-gray-300 text-xs">{m.avg_kw} kW</td>
                      <td className="px-6 py-4 font-mono text-xs text-amber-400">{m.outage_intervals} bins</td>
                      <td className="px-6 py-4">
                        <span className="px-2.5 py-1 text-[10px] font-bold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 uppercase tracking-wider">
                          Compliant
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: QUALITY GATES FUNNEL */}
      {activeTab === 'QUALITY' && (
        <div className="space-y-6">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h2 className="text-lg font-bold text-white mb-2">5-Stage Physical Ingestion Quality Gates</h2>
            <p className="text-gray-400 text-xs mb-6">
              Contract verification enforcing continuous span, bin coverage, outage caps, zero duplicates, and physical limits
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 mb-8">
              {[
                { name: 'Gate 1', title: 'Span >= 30d', rate: '92.8%', fails: '6 Failures' },
                { name: 'Gate 2', title: 'Coverage >= 70%', rate: '84.5%', fails: '13 Failures' },
                { name: 'Gate 3', title: 'Outage <= 40%', rate: '89.3%', fails: '9 Failures' },
                { name: 'Gate 4', title: '0 Duplicates', rate: '100.0%', fails: '0 Failures' },
                { name: 'Gate 5', title: 'Physical Bounds', rate: '98.8%', fails: '1 Failure' },
              ].map((g, idx) => (
                <div key={idx} className="bg-gray-800/60 border border-gray-700/70 p-4 rounded-xl text-center">
                  <span className="text-[10px] uppercase font-bold text-cyan-400 tracking-wider">{g.name}</span>
                  <div className="text-sm font-bold text-white mt-1">{g.title}</div>
                  <div className="text-xl font-extrabold text-emerald-400 mt-2">{g.rate}</div>
                  <span className="text-[10px] text-gray-500 mt-1 block">{g.fails}</span>
                </div>
              ))}
            </div>

            {/* Admission Funnel Visual */}
            <div className="space-y-3 bg-gray-950 p-6 rounded-xl border border-gray-800">
              <div>
                <div className="flex justify-between text-xs font-semibold text-gray-300 mb-1">
                  <span>Candidate Physical Meters Evaluated</span>
                  <span className="font-mono">84 Meters (100%)</span>
                </div>
                <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-gray-500 w-full rounded-full"></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs font-semibold text-cyan-400 mb-1">
                  <span>Admitted Compliant Ingestion Base (Passed All 5 Gates)</span>
                  <span className="font-mono">55 Meters (65.5% Admission Rate)</span>
                </div>
                <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-500 w-[65.5%] rounded-full shadow-lg shadow-cyan-500/50"></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs font-semibold text-rose-400 mb-1">
                  <span>Rejected Due to Sensor Gaps or Incomplete Bins</span>
                  <span className="font-mono">29 Meters (34.5% Rejected)</span>
                </div>
                <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-rose-500 w-[34.5%] rounded-full"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: DIURNAL LOAD & SOLAR DUCK CURVE */}
      {activeTab === 'DUCK_CURVE' && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h2 className="text-lg font-bold text-white">Diurnal Load & Solar Duck Curve</h2>
              <p className="text-xs text-gray-400 mt-0.5">
                Substation gross load demand vs rooftop solar photovoltaic generation offset (MW)
              </p>
            </div>
            <div className="flex items-center space-x-4 text-xs font-semibold">
              <span className="flex items-center space-x-1.5 text-cyan-400">
                <span className="w-3 h-3 bg-cyan-400 rounded-sm"></span>
                <span>Gross Demand</span>
              </span>
              <span className="flex items-center space-x-1.5 text-amber-400">
                <span className="w-3 h-3 bg-amber-400 rounded-sm"></span>
                <span>Solar PV Offset</span>
              </span>
              <span className="flex items-center space-x-1.5 text-emerald-400">
                <span className="w-3 h-3 bg-emerald-400 rounded-sm"></span>
                <span>Net Grid Load</span>
              </span>
            </div>
          </div>

          {/* SVG Multi-Line Chart */}
          <div className="w-full h-72 sm:h-80 relative bg-gray-950 p-4 rounded-xl border border-gray-800">
            <svg className="w-full h-full overflow-visible" viewBox="0 0 800 240" preserveAspectRatio="none">
              {/* Grid lines */}
              {[0, 60, 120, 180].map((y) => (
                <line key={y} x1="0" y1={y} x2="800" y2={y} stroke="#1f2937" strokeDasharray="3 3" />
              ))}

              {/* Gross Demand Line (Cyan) */}
              <path
                d="M 0 160 L 66 170 L 133 175 L 200 177 L 266 170 L 333 155 L 400 120 L 466 80 L 533 60 L 600 70 L 666 75 L 733 80 L 800 60"
                fill="none"
                stroke="#06b6d4"
                strokeWidth="3"
                strokeLinecap="round"
              />

              {/* Solar Generation (Amber) */}
              <path
                d="M 0 240 L 133 240 L 266 235 L 333 200 L 400 140 L 466 100 L 533 110 L 600 150 L 666 220 L 733 240 L 800 240"
                fill="none"
                stroke="#f59e0b"
                strokeWidth="2.5"
                strokeDasharray="4 4"
              />

              {/* Net Demand Curve (Emerald) */}
              <path
                d="M 0 160 L 66 170 L 133 175 L 200 177 L 266 170 L 333 170 L 400 185 L 466 180 L 533 145 L 600 120 L 666 90 L 733 80 L 800 60"
                fill="none"
                stroke="#10b981"
                strokeWidth="3.5"
                strokeLinecap="round"
              />
            </svg>

            <div className="flex justify-between text-xs text-gray-500 mt-2 font-mono border-t border-gray-800 pt-2">
              <span>00:00</span>
              <span>04:00</span>
              <span>08:00</span>
              <span>12:00 (Solar Peak)</span>
              <span>16:00</span>
              <span>19:00 (Evening Peak)</span>
              <span>23:45</span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
            <div className="p-4 bg-gray-800/60 rounded-xl border border-gray-700">
              <span className="text-gray-400">Peak Gross Load</span>
              <div className="text-xl font-bold text-cyan-400 mt-1">38.4 MW</div>
              <span className="text-[11px] text-gray-500">Occurs at 19:15 evening surge</span>
            </div>
            <div className="p-4 bg-gray-800/60 rounded-xl border border-gray-700">
              <span className="text-gray-400">Solar PV Peak Offset</span>
              <div className="text-xl font-bold text-amber-400 mt-1">19.2 MW</div>
              <span className="text-[11px] text-gray-500">Displaces 50% grid thermal draw</span>
            </div>
            <div className="p-4 bg-gray-800/60 rounded-xl border border-gray-700">
              <span className="text-gray-400">Duck Curve Belly Deficit</span>
              <div className="text-xl font-bold text-emerald-400 mt-1">14.5 MW Ramp</div>
              <span className="text-[11px] text-gray-500">Fast-response spinning reserve needed</span>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: RELIABILITY KPIS */}
      {activeTab === 'RELIABILITY' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-gray-900 border border-gray-800 p-5 rounded-xl">
              <span className="text-xs font-bold text-gray-400 uppercase">SAIDI</span>
              <div className="text-2xl font-bold text-cyan-400 mt-1">114.2 min</div>
              <span className="text-[11px] text-gray-500">System Avg Interruption Duration</span>
            </div>
            <div className="bg-gray-900 border border-gray-800 p-5 rounded-xl">
              <span className="text-xs font-bold text-gray-400 uppercase">SAIFI</span>
              <div className="text-2xl font-bold text-emerald-400 mt-1">1.18 events</div>
              <span className="text-[11px] text-gray-500">Interruption Frequency per Customer</span>
            </div>
            <div className="bg-gray-900 border border-gray-800 p-5 rounded-xl">
              <span className="text-xs font-bold text-gray-400 uppercase">CAIDI</span>
              <div className="text-2xl font-bold text-amber-400 mt-1">96.8 min</div>
              <span className="text-[11px] text-gray-500">Customer Restoration Time</span>
            </div>
            <div className="bg-gray-900 border border-gray-800 p-5 rounded-xl">
              <span className="text-xs font-bold text-gray-400 uppercase">ASAI Availability</span>
              <div className="text-2xl font-bold text-purple-400 mt-1">99.978%</div>
              <span className="text-[11px] text-purple-400/80">IEEE 1366 High Reliability Tier</span>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h2 className="text-base font-bold text-white mb-4">Grid Operational Stability & Efficiency</h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
              <div className="p-4 bg-gray-800/40 rounded-xl border border-gray-700/60">
                <span className="text-gray-400">Total Customer Connections</span>
                <div className="text-xl font-bold text-white mt-1">55,000</div>
                <span className="text-[11px] text-gray-500">Served by 55 admitted substations</span>
              </div>
              <div className="p-4 bg-gray-800/40 rounded-xl border border-gray-700/60">
                <span className="text-gray-400">Technical Distribution Loss</span>
                <div className="text-xl font-bold text-emerald-400 mt-1">5.4%</div>
                <span className="text-[11px] text-gray-500">Below 8% regulatory ceiling</span>
              </div>
              <div className="p-4 bg-gray-800/40 rounded-xl border border-gray-700/60">
                <span className="text-gray-400">Substation Load Factor</span>
                <div className="text-xl font-bold text-cyan-400 mt-1">0.68</div>
                <span className="text-[11px] text-gray-500">Optimal thermal reserve margin</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

