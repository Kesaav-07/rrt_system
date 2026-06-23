"""realtime package — re-exports public API from realtime.realtime."""
from realtime.realtime import (
    run_synthetic_tick_if_due,
    connection_status,
    STATUS_DISPLAY,
    get_alerts,
    load_patient_history,
    load_all_history,
)

__all__ = [
    "run_synthetic_tick_if_due",
    "connection_status",
    "STATUS_DISPLAY",
    "get_alerts",
    "load_patient_history",
    "load_all_history",
]
