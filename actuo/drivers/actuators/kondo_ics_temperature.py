"""KRS/ICS servo temperature table: raw setting value -> degrees Celsius.

The servo reports temperature as a thermistor *setting value*, not a
temperature: the mapping is monotonically **decreasing** and piecewise linear
with several different slopes, so it cannot be recovered from a formula. The
table is therefore data, reproduced verbatim from the hardware-validated
``rcb4`` implementation (``rcb4/temperature.py``), which in turn follows the
Kondo ICS servo specification.

Index 0 is not a temperature -- the firmware leaves the byte at 0 until it has
actually polled the servo -- so callers must treat 0 as "not measured" rather
than as 118 degrees. :func:`setting_value_to_celsius` enforces that.

Source: ``rcb4/temperature.py`` (github.com/iory/rcb4), see
``actuo/drivers/references.md``.
"""

from __future__ import annotations

from typing import Optional, Tuple

#: Degrees Celsius indexed by the servo's raw temperature setting value.
SETTING_VALUE_TO_CELSIUS: Tuple[float, ...] = (
    118.0, 117.0588, 116.4706, 115.8824, 115.2941, 114.7059, 114.1176,
    113.5294, 112.9412, 112.3529, 111.7647, 111.1765, 110.5882, 110.0,
    109.4118, 108.8235, 108.2353, 107.6471, 107.0588, 106.4706, 105.8824,
    105.2941, 104.7059, 104.1176, 103.5294, 102.9412, 102.3529, 101.7647,
    101.1765, 100.5882, 100.0, 99.4118, 98.8235, 98.2353, 97.6471, 97.0588,
    96.4706, 95.8824, 95.2941, 94.7059, 94.1176, 93.5294, 92.9412, 92.3529,
    91.7647, 91.1765, 90.5882, 90.0, 89.2308, 88.4615, 87.6923, 86.9231,
    86.1538, 85.3846, 84.6154, 83.8462, 83.0769, 82.3077, 81.5385, 80.7692,
    80.0, 79.3333, 78.6667, 78.0, 77.3333, 76.6667, 76.0, 75.3333, 74.6667,
    74.0, 73.3333, 72.6667, 72.0, 71.3333, 70.6667, 70.0, 69.1667, 68.3333,
    67.5, 66.6667, 65.8333, 65.0, 64.1667, 63.3333, 62.5, 61.6667, 60.8333,
    60.0, 59.1667, 58.3333, 57.5, 56.6667, 55.8333, 55.0, 54.1667, 53.3333,
    52.5, 51.6667, 50.8333, 50.0, 49.1667, 48.3333, 47.5, 46.6667, 45.8333,
    45.0, 44.1667, 43.3333, 42.5, 41.6667, 40.8333, 40.0, 39.1667, 38.3333,
    37.5, 36.6667, 35.8333, 35.0, 34.1667, 33.3333, 32.5, 31.6667, 30.8333,
    30.0, 29.1667, 28.3333, 27.5, 26.6667,
)


def setting_value_to_celsius(raw: int) -> Optional[float]:
    """Convert a raw temperature setting value to degrees Celsius.

    Parameters
    ----------
    raw : int
        The servo's raw temperature byte.

    Returns
    -------
    float or None
        The temperature, or ``None`` when ``raw`` is 0 -- meaning the firmware
        has not polled temperature (the read flag is off). Returning 118 degrees
        there would be a fabricated over-temperature reading.

    Raises
    ------
    IndexError
        If ``raw`` is outside the table.
    """
    if raw == 0:
        return None
    return SETTING_VALUE_TO_CELSIUS[raw]


def celsius_to_setting_value(celsius: float) -> int:
    """Nearest raw setting value for a temperature, for writing a limit."""
    best = 1
    best_error = abs(SETTING_VALUE_TO_CELSIUS[1] - celsius)
    for index in range(2, len(SETTING_VALUE_TO_CELSIUS)):
        error = abs(SETTING_VALUE_TO_CELSIUS[index] - celsius)
        if error < best_error:
            best, best_error = index, error
    return best
