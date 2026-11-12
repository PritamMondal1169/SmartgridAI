import React, { useState } from 'react';

export default function ConsumptionChart() {
  const [selectedMeter, setSelectedMeter] = useState('MTR_001');

  // Simulated 24-point 15-minute telemetry intervals
  const mockHours = Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, '0')}:00`);
  const mockSeries = [
    12.4, 11.2, 10.8, 10.5, 11.8, 14.5, 18.2, 22.8, 25.4, 24.1, 23.0, 22.5,
    21.8, 23.4, 26.2, 28.5, 31.2, 29.8, 27.4, 24.1, 20.5, 17.2, 14.8, 13.0,
  ];

  const maxVal = Math.max(...mockSeries);
  const minVal = Math.min(...mockSeries);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Telemetry Consumption Curve</h1>
          <p className="text-gray-400 text-sm mt-1">15-minute resampled active power (kW) profiles</p>
        </div>

        <div className="flex items-center space-x-3">
          <label className="text-xs text-gray-400 font-medium">Select Feeder:</label>
          <select
            value={selectedMeter}
            onChange={(e) => setSelectedMeter(e.target.value)}
            className="bg-gray-900 border border-gray-700 text-white text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:border-cyan-500"
          >
            <option value="MTR_001">Feeder A-01 (MTR_001 - Residential)</option>
            <option value="MTR_002">Feeder B-12 (MTR_002 - Commercial)</option>
            <option value="MTR_003">Feeder C-08 (MTR_003 - Industrial)</option>
          </select>
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 shadow-xl">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-6">
            <div>
              <div className="text-xs text-gray-400">Peak Load</div>
              <div className="text-xl font-bold text-cyan-400">{maxVal.toFixed(1)} kW</div>
            </div>
            <div>
              <div className="text-xs text-gray-400">Base Load</div>
              <div className="text-xl font-bold text-gray-200">{minVal.toFixed(1)} kW</div>
            </div>
            <div>
              <div className="text-xs text-gray-400">Average Demand</div>
              <div className="text-xl font-bold text-emerald-400">
                {(mockSeries.reduce((a, b) => a + b, 0) / mockSeries.length).toFixed(1)} kW
              </div>
            </div>
          </div>
          <div className="text-xs text-gray-500 font-mono">Sampling: 96 intervals / 24h</div>
        </div>

        {/* SVG Interactive Consumption Chart */}
        <div className="w-full h-64 sm:h-80 relative">
          <svg className="w-full h-full overflow-visible" viewBox="0 0 800 240" preserveAspectRatio="none">
            <defs>
              <linearGradient id="gridGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.35" />
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Horizontal Grid lines */}
            {[0, 60, 120, 180].map((y) => (
              <line key={y} x1="0" y1={y} x2="800" y2={y} stroke="#1f2937" strokeDasharray="3 3" />
            ))}

            {/* Area path */}
            <path
              d={`
                M 0 240
                ${mockSeries
                  .map((val, idx) => {
                    const x = (idx / (mockSeries.length - 1)) * 800;
                    const y = 240 - ((val - minVal + 2) / (maxVal - minVal + 5)) * 200;
                    return `L ${x} ${y}`;
                  })
                  .join(' ')}
                L 800 240 Z
              `}
              fill="url(#gridGrad)"
            />

            {/* Line path */}
            <path
              d={`
                M 0 ${240 - ((mockSeries[0] - minVal + 2) / (maxVal - minVal + 5)) * 200}
                ${mockSeries
                  .map((val, idx) => {
                    const x = (idx / (mockSeries.length - 1)) * 800;
                    const y = 240 - ((val - minVal + 2) / (maxVal - minVal + 5)) * 200;
                    return `L ${x} ${y}`;
                  })
                  .join(' ')}
              `}
              fill="none"
              stroke="#06b6d4"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Points */}
            {mockSeries.map((val, idx) => {
              const x = (idx / (mockSeries.length - 1)) * 800;
              const y = 240 - ((val - minVal + 2) / (maxVal - minVal + 5)) * 200;
              return (
                <circle
                  key={idx}
                  cx={x}
                  cy={y}
                  r="3.5"
                  className="fill-cyan-400 stroke-gray-900 stroke-2 hover:r-6 cursor-pointer transition-all"
                >
                  <title>{`${mockHours[idx]}: ${val} kW`}</title>
                </circle>
              );
            })}
          </svg>
        </div>

        {/* X Axis labels */}
        <div className="flex justify-between text-xs text-gray-500 mt-4 border-t border-gray-800 pt-2 font-mono">
          <span>00:00</span>
          <span>04:00</span>
          <span>08:00</span>
          <span>12:00</span>
          <span>16:00</span>
          <span>20:00</span>
          <span>23:45</span>
        </div>
      </div>
    </div>
  );
}

