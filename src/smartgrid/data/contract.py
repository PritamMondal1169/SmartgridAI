"""
SmartgridAI Physical Data Contracts and Conversion Functions.
Authoritative module for electrical physics conversions, outage definitions,
and telemetry validity rules.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import math

# --- Physical Constants ---
INTERVAL_MINUTES: float = 3.0
CONVERSION_FACTOR: float = 20.0  # 60.0 / INTERVAL_MINUTES (1 hr = 20 * 3 min)
RESAMPLE_INTERVAL_MINUTES: int = 15

# Grid outage detection thresholds
VOLTAGE_OUTAGE_THRESHOLD_V: float = 5.0
CURRENT_OUTAGE_THRESHOLD_A: float = 0.01

# Quality Gate Thresholds
MIN_SPAN_DAYS: float = 30.0
MIN_BIN_COVERAGE: float = 0.70  # Yields 55 admitted out of 84 (65.5% admission rate)
MAX_OUTAGE_FRACTION: float = 0.40

# Electrical physical bounds for Indian distribution grids
VOLTAGE_MIN_V: float = 0.0
VOLTAGE_MAX_V: float = 650.0  # Nominal 230V single-phase with severe over-voltage spikes up to 650V
VOLTAGE_STANDARD_MAX_V: float = 320.0  # Standard single-phase distribution upper limit
CURRENT_MIN_A: float = 0.0
CURRENT_MAX_A: float = 150.0  # Nominal 0-100A with surge capacity up to 150A
CURRENT_STANDARD_MAX_A: float = 100.0
FREQ_MIN_HZ: float = 0.0
FREQ_MAX_HZ: float = 150.0  # Nominal 50Hz with transient spikes up to 150Hz
FREQ_STANDARD_MAX_HZ: float = 55.0


@dataclass(frozen=True)
class PhysicalReading:
    """Represents a single raw physical meter observation."""
    timestamp: datetime  # UTC normalized
    meter_id: str
    energy_kwh: float
    voltage_v: float
    current_a: float
    freq_hz: float
    power_kw: Optional[float]
    is_outage: bool

    def to_dict(self) -> Dict[str, Any]:
        """Converts reading to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "meter_id": self.meter_id,
            "energy_kwh": self.energy_kwh,
            "voltage_v": self.voltage_v,
            "current_a": self.current_a,
            "freq_hz": self.freq_hz,
            "power_kw": self.power_kw,
            "is_outage": self.is_outage,
        }


@dataclass(frozen=True)
class ResampledReading:
    """Represents a 15-minute resampled telemetry interval."""
    timestamp: datetime  # Interval start (UTC)
    meter_id: str
    power_kw: Optional[float]
    voltage_v: float
    current_a: float
    freq_hz: float
    is_outage: bool
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Converts resampled interval to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "meter_id": self.meter_id,
            "power_kw": self.power_kw,
            "voltage_v": self.voltage_v,
            "current_a": self.current_a,
            "freq_hz": self.freq_hz,
            "is_outage": self.is_outage,
            "sample_count": self.sample_count,
        }


def compute_active_power(energy_kwh: float, interval_minutes: float = INTERVAL_MINUTES) -> float:
    """
    Computes active power in kW from energy in kWh.
    
    Formula: P_kW = energy_kwh * (60.0 / interval_minutes).
    For standard 3-minute intervals: P_kW = energy_kwh * 20.0.
    
    Args:
        energy_kwh: Active energy consumed in kWh during the interval.
        interval_minutes: Duration of interval in minutes (default 3.0).
        
    Returns:
        Active power in kW.
        
    Raises:
        ValueError: If interval_minutes is non-positive or energy_kwh is negative.
    """
    if interval_minutes <= 0.0:
        raise ValueError(f"interval_minutes must be strictly positive, got {interval_minutes}")
    if energy_kwh < 0.0:
        raise ValueError(f"energy_kwh cannot be negative, got {energy_kwh}")
    return energy_kwh * (60.0 / interval_minutes)


def is_outage_reading(voltage_v: float, current_a: float) -> bool:
    """
    Determines whether telemetry indicates a distribution grid supply outage.
    
    Condition: V <= 5.0 V or I <= 0.01 A.
    
    Args:
        voltage_v: Phase voltage in Volts.
        current_a: Phase current in Amperes.
        
    Returns:
        True if grid supply is cut off, False otherwise.
    """
    return (voltage_v <= VOLTAGE_OUTAGE_THRESHOLD_V) or (current_a <= CURRENT_OUTAGE_THRESHOLD_A)


def apply_physical_contract(
    energy_kwh: float,
    voltage_v: float,
    current_a: float,
    interval_minutes: float = INTERVAL_MINUTES
) -> Tuple[Optional[float], bool]:
    """
    Applies physical power conversion and grid outage masking:
    - If outage condition met (V <= 5.0V or I <= 0.01A):
        power_kw = None (NaN representation in pure Python/JSON), is_outage = True
    - Normal operation:
        power_kw = compute_active_power(energy_kwh, interval_minutes), is_outage = False
        
    Args:
        energy_kwh: Active energy in kWh.
        voltage_v: Voltage in Volts.
        current_a: Current in Amperes.
        interval_minutes: Duration in minutes.
        
    Returns:
        Tuple of (power_kw, is_outage).
    """
    outage = is_outage_reading(voltage_v, current_a)
    if outage:
        return None, True
    power_kw = compute_active_power(energy_kwh, interval_minutes)
    return power_kw, False


def validate_reading_bounds(
    energy_kwh: float,
    voltage_v: float,
    current_a: float,
    freq_hz: float,
    max_voltage: float = VOLTAGE_MAX_V,
    max_current: float = CURRENT_MAX_A,
    max_freq: float = FREQ_MAX_HZ
) -> List[str]:
    """
    Validates physical parameters against electrical operating envelopes.
    
    Args:
        energy_kwh: Active energy in kWh.
        voltage_v: Voltage in Volts.
        current_a: Current in Amperes.
        freq_hz: Frequency in Hertz.
        max_voltage: Maximum plausible voltage limit.
        max_current: Maximum plausible current limit.
        max_freq: Maximum plausible frequency limit.
        
    Returns:
        List of violation codes. Empty list if reading is physically valid.
    """
    violations: List[str] = []
    if energy_kwh < 0.0:
        violations.append("negative_energy")
    if voltage_v < VOLTAGE_MIN_V or voltage_v > max_voltage:
        violations.append("voltage_out_of_bounds")
    if current_a < CURRENT_MIN_A or current_a > max_current:
        violations.append("current_out_of_bounds")
    if freq_hz < FREQ_MIN_HZ or freq_hz > max_freq:
        violations.append("frequency_out_of_bounds")
    return violations
