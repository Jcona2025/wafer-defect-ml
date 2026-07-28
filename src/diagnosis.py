"""Signature -> root-cause guidance shown by the triage tool."""

ROOT_CAUSES = {
    "Edge-Ring": {
        "direction": "Chamber uniformity / edge processes",
        "first_checks": "Edge exclusion settings, clamp/chuck contact, chamber matching on the last etch/dep steps",
    },
    "Edge-Loc": {
        "direction": "Single-side edge process issue",
        "first_checks": "Orientation vs. notch, single-chamber edge non-uniformity, handling contact points",
    },
    "Center": {
        "direction": "Center-of-wafer non-uniformity",
        "first_checks": "Dispense/spin steps, center-to-edge etch rate delta, showerplate condition, quartz condition",
    },
    "Donut": {
        "direction": "Radial process non-uniformity",
        "first_checks": "Mid-radius rate anomalies, temperature profile, plasma density ring effects",
    },
    "Scratch": {
        "direction": "Wafer handling / transfer damage",
        "first_checks": "Robot handoffs, pin marks, cassette/FOUP contact on tools this lot visited",
    },
    "Loc": {
        "direction": "Localized event — particle or contact",
        "first_checks": "Recent PM on tools in route, single-point chuck defects, particle excursion on inline monitors",
    },
    "Random": {
        "direction": "Baseline defectivity, no single assignable cause",
        "first_checks": "Overall defect density trend vs. baseline; no targeted action from map alone",
    },
    "Near-full": {
        "direction": "Gross process failure",
        "first_checks": "Wrong recipe, skipped step, misprocessed lot — check route history first",
    },
    "none": {
        "direction": "Healthy baseline",
        "first_checks": "No action needed",
    },
}
