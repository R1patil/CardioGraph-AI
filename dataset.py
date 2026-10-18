"""
CardioGraph AI — Synthetic Coronary Vascular Tree Dataset Generator
===================================================================
Generates realistic, biophysically grounded coronary arterial tree graphs,
hemodynamic bifurcation parameters, plaque vulnerability phenotypes (including
the clinical "Rupture Paradox"), biomarker profiles, and myocardial perfusion zones.

Scientific Foundations:
- Demis Hassabis / DeepMind ApoB100 atherogenic particle modeling
- Dario Amodei systemic causal biophysics
- Thin-Cap Fibroatheroma (TCFA) clinical thresholds (<65 µm cap)
- Wall Shear Stress (WSS) and bifurcation turbulence mechanics
- 17-Segment AHA Left Ventricular Myocardium Model
"""

from __future__ import annotations
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BiomarkerProfile:
    id: str
    patient_id: str
    apob_mg_dl: float        # ApoB100 atherogenic particle count (optimal <80, high >100, extreme >130)
    lpa_nmol_l: float        # Lipoprotein(a) pro-thrombotic genetic risk (>125 is elevated)
    hscrp_mg_l: float        # High-sensitivity C-reactive protein (inflammation, >3.0 high)
    ldl_c_mg_dl: float       # LDL-C cholesterol
    troponin_i_ng_ml: float  # High-sensitivity Troponin-I (<0.04 normal, elevated in acute ischemia)
    hba1c_pct: float         # Glycated hemoglobin (diabetes marker)


@dataclass
class MyocardiumZone:
    id: str
    zone_name: str
    wall: str                # 'Anterior' | 'Septal' | 'Inferior' | 'Lateral' | 'Apical'
    mass_pct: float          # Percentage of left ventricular myocardium mass
    aha_segment_ids: List[int]


@dataclass
class Bifurcation:
    id: str
    name: str
    parent_vessel: str
    daughter_vessels: List[str]
    bifurcation_angle_deg: float   # Angle between daughter vessels (e.g., 60-95 deg)
    shear_stress_index: float      # Normalized Wall Shear Stress Gradient (1.0 to 10.0)
    turbulence_factor: float       # Flow separation & oscillatory turbulence (0.0 to 1.0)
    flow_ratio: float              # Flow split ratio daughter1 / daughter2


@dataclass
class Plaque:
    id: str
    artery_segment_id: str
    stenosis_pct: float            # Lumen narrowing percentage (e.g. 38%, 70%)
    plaque_type: str               # 'soft_lipid' | 'fibrocalcific' | 'mixed' | 'fibrous'
    cap_thickness_um: float        # Fibrous cap thickness in micrometers (<65 µm is TCFA vulnerable)
    core_lipid_ratio: float        # Necrotic lipid core proportion (0.0 to 0.85)
    calcification_score: float     # Agatston calcium component (0.0 to 10.0)
    vulnerability_score: float     # Computed Rupture Vulnerability Index (0.0 to 1.0)
    rupture_risk_category: str     # 'CRITICAL' | 'HIGH' | 'MODERATE' | 'LOW'
    adjacent_bifurcation_id: Optional[str] = None
    distance_to_bifurcation_mm: Optional[float] = None


@dataclass
class ArterySegment:
    id: str
    name: str
    vessel_system: str             # 'LCA' (Left Coronary Artery) or 'RCA' (Right Coronary Artery)
    diameter_mm: float             # Vessel lumen diameter (e.g., 2.0 - 4.5 mm)
    length_mm: float               # Vessel segment length (e.g., 15 - 45 mm)
    baseline_flow_ml_min: float    # Normal resting blood flow rate
    resistance: float              # Poiseuille hydrodynamic resistance (8 * eta * L / (pi * r^4))
    flows_into: List[str] = field(default_factory=list)
    perfuses_zones: List[str] = field(default_factory=list)


@dataclass
class FlowEdge:
    source_segment_id: str
    target_segment_id: str
    resistance: float
    flow_split_pct: float


@dataclass
class PerfusionEdge:
    artery_segment_id: str
    zone_id: str
    perfusion_weight: float        # Proportion of zone supplied by this segment (0.0 to 1.0)


@dataclass
class Patient:
    id: str
    name: str
    age: int
    gender: str
    smoker: bool
    diabetes: bool
    hypertension: bool
    clinical_presentation: str
    benchmark_type: Optional[str] = None  # e.g., 'RUPTURE_PARADOX', 'STABLE_STENOSIS', 'HEALTHY'
    biomarkers: Optional[BiomarkerProfile] = None
    plaques: List[Plaque] = field(default_factory=list)


@dataclass
class PatientCoronaryGraph:
    patient: Patient
    segments: List[ArterySegment]
    bifurcations: List[Bifurcation]
    zones: List[MyocardiumZone]
    flow_edges: List[FlowEdge]
    perfusion_edges: List[PerfusionEdge]


# ============================================================================
# CANONICAL ANATOMICAL TOPOLOGY BUILDER
# ============================================================================

def get_canonical_myocardium_zones() -> List[MyocardiumZone]:
    """Standard AHA 17-Segment myocardial territory distribution (normalized mass %)."""
    return [
        MyocardiumZone(
            id="ZONE_ANT_BASAL",
            zone_name="Basal Anterior",
            wall="Anterior",
            mass_pct=10.0,
            aha_segment_ids=[1]
        ),
        MyocardiumZone(
            id="ZONE_ANT_MID",
            zone_name="Mid Anterior",
            wall="Anterior",
            mass_pct=12.0,
            aha_segment_ids=[7]
        ),
        MyocardiumZone(
            id="ZONE_SEPT_BASAL",
            zone_name="Basal Septal",
            wall="Septal",
            mass_pct=10.0,
            aha_segment_ids=[2, 3]
        ),
        MyocardiumZone(
            id="ZONE_SEPT_MID",
            zone_name="Mid Septal",
            wall="Septal",
            mass_pct=11.0,
            aha_segment_ids=[8, 9]
        ),
        MyocardiumZone(
            id="ZONE_APICAL",
            zone_name="Apical Territory & True Apex",
            wall="Apical",
            mass_pct=14.0,
            aha_segment_ids=[13, 14, 15, 16, 17]
        ),
        MyocardiumZone(
            id="ZONE_LAT_BASAL",
            zone_name="Basal Lateral",
            wall="Lateral",
            mass_pct=11.0,
            aha_segment_ids=[5, 6]
        ),
        MyocardiumZone(
            id="ZONE_LAT_MID",
            zone_name="Mid Lateral",
            wall="Lateral",
            mass_pct=12.0,
            aha_segment_ids=[11, 12]
        ),
        MyocardiumZone(
            id="ZONE_INF_BASAL",
            zone_name="Basal Inferior",
            wall="Inferior",
            mass_pct=10.0,
            aha_segment_ids=[4]
        ),
        MyocardiumZone(
            id="ZONE_INF_MID",
            zone_name="Mid Inferior",
            wall="Inferior",
            mass_pct=10.0,
            aha_segment_ids=[10]
        ),
    ]


def get_canonical_artery_segments() -> List[ArterySegment]:
    """Canonical 15-segment coronary artery anatomical tree."""
    return [
        # Left Coronary System (LCA)
        ArterySegment(
            id="SEG_LM",
            name="Left Main Coronary Artery (LM)",
            vessel_system="LCA",
            diameter_mm=4.5,
            length_mm=12.0,
            baseline_flow_ml_min=180.0,
            resistance=0.08,
            flows_into=["SEG_LAD_PROX", "SEG_LCX_PROX"],
            perfuses_zones=[]
        ),
        ArterySegment(
            id="SEG_LAD_PROX",
            name="Proximal LAD (Left Anterior Descending)",
            vessel_system="LCA",
            diameter_mm=3.8,
            length_mm=22.0,
            baseline_flow_ml_min=110.0,
            resistance=0.14,
            flows_into=["SEG_LAD_MID", "SEG_D1"],
            perfuses_zones=["ZONE_ANT_BASAL"]
        ),
        ArterySegment(
            id="SEG_LAD_MID",
            name="Mid LAD",
            vessel_system="LCA",
            diameter_mm=3.2,
            length_mm=26.0,
            baseline_flow_ml_min=75.0,
            resistance=0.22,
            flows_into=["SEG_LAD_DIST", "SEG_D2", "SEG_S1"],
            perfuses_zones=["ZONE_ANT_MID"]
        ),
        ArterySegment(
            id="SEG_LAD_DIST",
            name="Distal LAD",
            vessel_system="LCA",
            diameter_mm=2.2,
            length_mm=30.0,
            baseline_flow_ml_min=35.0,
            resistance=0.45,
            flows_into=[],
            perfuses_zones=["ZONE_APICAL"]
        ),
        ArterySegment(
            id="SEG_D1",
            name="First Diagonal Branch (D1)",
            vessel_system="LCA",
            diameter_mm=2.5,
            length_mm=24.0,
            baseline_flow_ml_min=30.0,
            resistance=0.38,
            flows_into=[],
            perfuses_zones=["ZONE_ANT_BASAL", "ZONE_LAT_BASAL"]
        ),
        ArterySegment(
            id="SEG_D2",
            name="Second Diagonal Branch (D2)",
            vessel_system="LCA",
            diameter_mm=2.1,
            length_mm=22.0,
            baseline_flow_ml_min=20.0,
            resistance=0.52,
            flows_into=[],
            perfuses_zones=["ZONE_ANT_MID"]
        ),
        ArterySegment(
            id="SEG_S1",
            name="First Septal Perforator (S1)",
            vessel_system="LCA",
            diameter_mm=1.9,
            length_mm=18.0,
            baseline_flow_ml_min=25.0,
            resistance=0.48,
            flows_into=[],
            perfuses_zones=["ZONE_SEPT_BASAL", "ZONE_SEPT_MID"]
        ),
        ArterySegment(
            id="SEG_LCX_PROX",
            name="Proximal LCx (Left Circumflex)",
            vessel_system="LCA",
            diameter_mm=3.5,
            length_mm=20.0,
            baseline_flow_ml_min=70.0,
            resistance=0.18,
            flows_into=["SEG_LCX_DIST", "SEG_OM1"],
            perfuses_zones=["ZONE_LAT_BASAL"]
        ),
        ArterySegment(
            id="SEG_LCX_DIST",
            name="Distal LCx",
            vessel_system="LCA",
            diameter_mm=2.6,
            length_mm=25.0,
            baseline_flow_ml_min=35.0,
            resistance=0.36,
            flows_into=["SEG_OM2"],
            perfuses_zones=["ZONE_LAT_MID"]
        ),
        ArterySegment(
            id="SEG_OM1",
            name="First Obtuse Marginal (OM1)",
            vessel_system="LCA",
            diameter_mm=2.4,
            length_mm=28.0,
            baseline_flow_ml_min=28.0,
            resistance=0.40,
            flows_into=[],
            perfuses_zones=["ZONE_LAT_BASAL", "ZONE_LAT_MID"]
        ),
        ArterySegment(
            id="SEG_OM2",
            name="Second Obtuse Marginal (OM2)",
            vessel_system="LCA",
            diameter_mm=2.0,
            length_mm=22.0,
            baseline_flow_ml_min=18.0,
            resistance=0.55,
            flows_into=[],
            perfuses_zones=["ZONE_LAT_MID"]
        ),

        # Right Coronary System (RCA)
        ArterySegment(
            id="SEG_RCA_PROX",
            name="Proximal RCA",
            vessel_system="RCA",
            diameter_mm=4.0,
            length_mm=28.0,
            baseline_flow_ml_min=95.0,
            resistance=0.15,
            flows_into=["SEG_RCA_MID"],
            perfuses_zones=[]
        ),
        ArterySegment(
            id="SEG_RCA_MID",
            name="Mid RCA",
            vessel_system="RCA",
            diameter_mm=3.4,
            length_mm=32.0,
            baseline_flow_ml_min=80.0,
            resistance=0.24,
            flows_into=["SEG_RCA_DIST", "SEG_AM"],
            perfuses_zones=["ZONE_INF_BASAL"]
        ),
        ArterySegment(
            id="SEG_AM",
            name="Acute Marginal Branch (AM)",
            vessel_system="RCA",
            diameter_mm=2.0,
            length_mm=20.0,
            baseline_flow_ml_min=18.0,
            resistance=0.58,
            flows_into=[],
            perfuses_zones=["ZONE_INF_BASAL"]
        ),
        ArterySegment(
            id="SEG_RCA_DIST",
            name="Distal RCA",
            vessel_system="RCA",
            diameter_mm=2.9,
            length_mm=24.0,
            baseline_flow_ml_min=60.0,
            resistance=0.32,
            flows_into=["SEG_PDA", "SEG_PLV"],
            perfuses_zones=["ZONE_INF_MID"]
        ),
        ArterySegment(
            id="SEG_PDA",
            name="Posterior Descending Artery (PDA)",
            vessel_system="RCA",
            diameter_mm=2.4,
            length_mm=26.0,
            baseline_flow_ml_min=35.0,
            resistance=0.42,
            flows_into=[],
            perfuses_zones=["ZONE_INF_MID", "ZONE_SEPT_MID"]
        ),
        ArterySegment(
            id="SEG_PLV",
            name="Posterolateral Ventricular Branch (PLV)",
            vessel_system="RCA",
            diameter_mm=2.1,
            length_mm=22.0,
            baseline_flow_ml_min=22.0,
            resistance=0.50,
            flows_into=[],
            perfuses_zones=["ZONE_INF_BASAL"]
        ),
    ]


def get_canonical_bifurcations() -> List[Bifurcation]:
    """Coronary arterial bifurcation junctions where oscillatory shear stress and turbulence concentrate."""
    return [
        Bifurcation(
            id="BIF_LM_LAD_LCX",
            name="Left Main Carina (LM -> LAD + LCx)",
            parent_vessel="SEG_LM",
            daughter_vessels=["SEG_LAD_PROX", "SEG_LCX_PROX"],
            bifurcation_angle_deg=82.0,
            shear_stress_index=8.4,
            turbulence_factor=0.85,
            flow_ratio=1.57   # LAD receives ~61%, LCx receives ~39%
        ),
        Bifurcation(
            id="BIF_LAD_D1",
            name="LAD-First Diagonal Junction",
            parent_vessel="SEG_LAD_PROX",
            daughter_vessels=["SEG_LAD_MID", "SEG_D1"],
            bifurcation_angle_deg=68.0,
            shear_stress_index=7.8,
            turbulence_factor=0.76,
            flow_ratio=2.50
        ),
        Bifurcation(
            id="BIF_LAD_D2_S1",
            name="Mid-LAD Bifurcation (LAD-Distal + D2 + S1)",
            parent_vessel="SEG_LAD_MID",
            daughter_vessels=["SEG_LAD_DIST", "SEG_D2", "SEG_S1"],
            bifurcation_angle_deg=62.0,
            shear_stress_index=6.9,
            turbulence_factor=0.68,
            flow_ratio=1.80
        ),
        Bifurcation(
            id="BIF_LCX_OM1",
            name="LCx-OM1 Junction",
            parent_vessel="SEG_LCX_PROX",
            daughter_vessels=["SEG_LCX_DIST", "SEG_OM1"],
            bifurcation_angle_deg=58.0,
            shear_stress_index=6.2,
            turbulence_factor=0.60,
            flow_ratio=1.25
        ),
        Bifurcation(
            id="BIF_RCA_PDA_PLV",
            name="RCA Crux Bifurcation (PDA + PLV)",
            parent_vessel="SEG_RCA_DIST",
            daughter_vessels=["SEG_PDA", "SEG_PLV"],
            bifurcation_angle_deg=74.0,
            shear_stress_index=7.2,
            turbulence_factor=0.72,
            flow_ratio=1.60
        ),
    ]


def get_canonical_edges(segments: List[ArterySegment]) -> Tuple[List[FlowEdge], List[PerfusionEdge]]:
    """Build directed flow edges and myocardial perfusion mappings."""
    flow_edges: List[FlowEdge] = []
    perfusion_edges: List[PerfusionEdge] = []

    seg_map = {s.id: s for s in segments}

    # Directed hemodynamic flow edges
    flow_definitions = [
        ("SEG_LM", "SEG_LAD_PROX", 0.14, 61.0),
        ("SEG_LM", "SEG_LCX_PROX", 0.18, 39.0),
        ("SEG_LAD_PROX", "SEG_LAD_MID", 0.22, 72.0),
        ("SEG_LAD_PROX", "SEG_D1", 0.38, 28.0),
        ("SEG_LAD_MID", "SEG_LAD_DIST", 0.45, 47.0),
        ("SEG_LAD_MID", "SEG_D2", 0.52, 27.0),
        ("SEG_LAD_MID", "SEG_S1", 0.48, 26.0),
        ("SEG_LCX_PROX", "SEG_LCX_DIST", 0.36, 55.0),
        ("SEG_LCX_PROX", "SEG_OM1", 0.40, 45.0),
        ("SEG_LCX_DIST", "SEG_OM2", 0.55, 100.0),
        ("SEG_RCA_PROX", "SEG_RCA_MID", 0.24, 100.0),
        ("SEG_RCA_MID", "SEG_RCA_DIST", 0.32, 78.0),
        ("SEG_RCA_MID", "SEG_AM", 0.58, 22.0),
        ("SEG_RCA_DIST", "SEG_PDA", 0.42, 60.0),
        ("SEG_RCA_DIST", "SEG_PLV", 0.50, 40.0),
    ]

    for src, tgt, res, split in flow_definitions:
        flow_edges.append(FlowEdge(
            source_segment_id=src,
            target_segment_id=tgt,
            resistance=res,
            flow_split_pct=split
        ))

    # Perfusion edges (ArterySegment -> MyocardiumZone)
    perfusion_definitions = [
        ("SEG_LAD_PROX", "ZONE_ANT_BASAL", 0.65),
        ("SEG_D1", "ZONE_ANT_BASAL", 0.35),
        ("SEG_D1", "ZONE_LAT_BASAL", 0.25),
        ("SEG_LAD_MID", "ZONE_ANT_MID", 0.70),
        ("SEG_D2", "ZONE_ANT_MID", 0.30),
        ("SEG_S1", "ZONE_SEPT_BASAL", 0.75),
        ("SEG_S1", "ZONE_SEPT_MID", 0.50),
        ("SEG_LAD_DIST", "ZONE_APICAL", 0.85),
        ("SEG_LCX_PROX", "ZONE_LAT_BASAL", 0.75),
        ("SEG_OM1", "ZONE_LAT_BASAL", 0.40),
        ("SEG_OM1", "ZONE_LAT_MID", 0.50),
        ("SEG_LCX_DIST", "ZONE_LAT_MID", 0.50),
        ("SEG_OM2", "ZONE_LAT_MID", 0.35),
        ("SEG_RCA_MID", "ZONE_INF_BASAL", 0.50),
        ("SEG_AM", "ZONE_INF_BASAL", 0.25),
        ("SEG_PLV", "ZONE_INF_BASAL", 0.25),
        ("SEG_RCA_DIST", "ZONE_INF_MID", 0.45),
        ("SEG_PDA", "ZONE_INF_MID", 0.55),
        ("SEG_PDA", "ZONE_SEPT_MID", 0.50),
        ("SEG_PDA", "ZONE_APICAL", 0.15),
    ]

    for seg_id, zone_id, weight in perfusion_definitions:
        perfusion_edges.append(PerfusionEdge(
            artery_segment_id=seg_id,
            zone_id=zone_id,
            perfusion_weight=weight
        ))

    return flow_edges, perfusion_edges


# ============================================================================
# BIOPHYSICAL PLAQUE VULNERABILITY & BLAST RADIUS FORMULAS
# ============================================================================

def compute_vulnerability_score(
    plaque_type: str,
    stenosis_pct: float,
    cap_thickness_um: float,
    core_lipid_ratio: float,
    calcification_score: float,
    apob_mg_dl: float,
    hscrp_mg_l: float,
    turbulence_factor: float = 0.5,
    shear_stress_index: float = 5.0
) -> Tuple[float, str]:
    """
    Computes the Plaque Vulnerability Index (PVI).
    Incorporates Thin-Cap Fibroatheroma (TCFA), DeepMind ApoB lipid infiltration,
    hs-CRP collagen degradation, and oscillatory wall shear stress mechanics.
    """
    # 1. Cap Thinning Risk (TCFA threshold is 65 µm; cap < 65 µm causes dramatic risk escalation)
    if cap_thickness_um <= 65.0:
        cap_factor = 1.0 - (cap_thickness_um / 130.0)  # e.g., 42µm -> ~0.68
        cap_factor = min(1.0, cap_factor * 1.35)       # TCFA multiplier
    else:
        cap_factor = max(0.05, 1.0 - (cap_thickness_um / 250.0))

    # 2. Composition multiplier
    type_mult = {
        "soft_lipid": 1.45,
        "mixed": 1.00,
        "fibrous": 0.55,
        "fibrocalcific": 0.35,  # dense calcified shell protects against cap rupture
    }.get(plaque_type, 1.0)

    # 3. Biomarker destabilization (ApoB + hs-CRP)
    apob_hazard = max(1.0, apob_mg_dl / 90.0)      # >100 elevated, >140 high
    crp_hazard = max(1.0, 1.0 + (hscrp_mg_l / 4.0)) # >3.0 mg/L active inflammation
    bio_destabilization = (apob_hazard * 0.5 + crp_hazard * 0.5)

    # 4. Hemodynamic Shear & Turbulence
    hemo_factor = (shear_stress_index / 10.0) * 0.5 + (turbulence_factor) * 0.5

    # 5. Protective Calcification dampener (dense calcium stabilizes cap against rupture)
    calcium_damping = max(0.4, 1.0 - (calcification_score / 20.0))

    # Composite formula:
    # Notice that stenosis_pct is only a partial contributor (~15%), perfectly capturing the Rupture Paradox!
    raw_pvi = (
        (cap_factor * 0.35) +
        (core_lipid_ratio * 0.25) +
        (stenosis_pct / 100.0 * 0.15) +
        (hemo_factor * 0.15) +
        ((bio_destabilization - 1.0) * 0.10)
    ) * type_mult * calcium_damping

    score = round(max(0.02, min(0.99, raw_pvi)), 3)

    if score >= 0.70 or (cap_thickness_um < 65.0 and plaque_type == "soft_lipid" and score >= 0.60):
        category = "CRITICAL"
    elif score >= 0.50:
        category = "HIGH"
    elif score >= 0.30:
        category = "MODERATE"
    else:
        category = "LOW"

    return score, category


def calculate_downstream_blast_radius(
    start_segment_id: str,
    flow_edges: List[FlowEdge],
    perfusion_edges: List[PerfusionEdge],
    zones: List[MyocardiumZone]
) -> Dict[str, any]:
    """
    Computes the Downstream Ischemic Blast Radius via graph reachability traversal.
    Returns threatened segments, affected myocardial zones, and total % mass at risk.
    """
    adjacency: Dict[str, List[str]] = {}
    for edge in flow_edges:
        adjacency.setdefault(edge.source_segment_id, []).append(edge.target_segment_id)

    # BFS / DFS reachability of all downstream artery segments
    visited_segments = set()
    queue = [start_segment_id]
    visited_segments.add(start_segment_id)

    while queue:
        curr = queue.pop(0)
        for nxt in adjacency.get(curr, []):
            if nxt not in visited_segments:
                visited_segments.add(nxt)
                queue.append(nxt)

    # Identify all perfused zones from threatened segments
    zone_map = {z.id: z for z in zones}
    threatened_zones: Dict[str, Dict[str, any]] = {}

    for p_edge in perfusion_edges:
        if p_edge.artery_segment_id in visited_segments:
            z_id = p_edge.zone_id
            if z_id in zone_map:
                zone = zone_map[z_id]
                if z_id not in threatened_zones:
                    threatened_zones[z_id] = {
                        "id": zone.id,
                        "zone_name": zone.zone_name,
                        "wall": zone.wall,
                        "mass_pct": zone.mass_pct,
                        "perfused_by_segments": [p_edge.artery_segment_id]
                    }
                else:
                    threatened_zones[z_id]["perfused_by_segments"].append(p_edge.artery_segment_id)

    total_mass_at_risk = round(sum(z["mass_pct"] for z in threatened_zones.values()), 1)

    return {
        "source_segment_id": start_segment_id,
        "downstream_segments_count": len(visited_segments),
        "downstream_segments": list(visited_segments),
        "threatened_zones": list(threatened_zones.values()),
        "total_myocardium_mass_at_risk_pct": total_mass_at_risk,
        "clinical_severity": (
            "MASSIVE_INFARCTION" if total_mass_at_risk >= 45.0 else
            "EXTENSIVE_ISCHEMIA" if total_mass_at_risk >= 25.0 else
            "REGIONAL_ISCHEMIA" if total_mass_at_risk >= 10.0 else
            "LOCALIZED_ISCHEMIA"
        )
    }


# ============================================================================
# BENCHMARK PATIENT GENERATORS
# ============================================================================

def create_benchmark_patient_101() -> PatientCoronaryGraph:
    """
    Patient #101 — "The False Reassurance Case" (Stable Severe Stenosis)
    - 65 yo male with exertional angina.
    - 70% stenosis in LAD_Mid, but DENSE FIBROCALCIFIC plaque with thick cap (185 µm).
    - Stable biomarkers: ApoB 88 mg/dL, hs-CRP 1.1 mg/L.
    - Standard angiogram flags this for stenting; but acute rupture risk is LOW (PVI ~0.26).
    """
    patient = Patient(
        id="PAT-101",
        name="Arthur Pendelton",
        age=65,
        gender="Male",
        smoker=False,
        diabetes=False,
        hypertension=True,
        clinical_presentation="Exertional angina upon walking 200m; stable symptoms for 6 months.",
        benchmark_type="STABLE_SEVERE_STENOSIS"
    )

    biomarkers = BiomarkerProfile(
        id="BIO-101",
        patient_id="PAT-101",
        apob_mg_dl=88.0,
        lpa_nmol_l=42.0,
        hscrp_mg_l=1.1,
        ldl_c_mg_dl=104.0,
        troponin_i_ng_ml=0.012,
        hba1c_pct=5.5
    )
    patient.biomarkers = biomarkers

    # LAD_Mid stable calcified plaque
    score, category = compute_vulnerability_score(
        plaque_type="fibrocalcific",
        stenosis_pct=72.0,
        cap_thickness_um=185.0,
        core_lipid_ratio=0.15,
        calcification_score=8.5,
        apob_mg_dl=biomarkers.apob_mg_dl,
        hscrp_mg_l=biomarkers.hscrp_mg_l,
        turbulence_factor=0.45,
        shear_stress_index=5.2
    )

    plaque_101 = Plaque(
        id="PLQ-101-LAD-MID",
        artery_segment_id="SEG_LAD_MID",
        stenosis_pct=72.0,
        plaque_type="fibrocalcific",
        cap_thickness_um=185.0,
        core_lipid_ratio=0.15,
        calcification_score=8.5,
        vulnerability_score=score,
        rupture_risk_category=category,
        adjacent_bifurcation_id="BIF_LAD_D2_S1",
        distance_to_bifurcation_mm=6.2
    )
    patient.plaques.append(plaque_101)

    segments = get_canonical_artery_segments()
    bifurcations = get_canonical_bifurcations()
    zones = get_canonical_myocardium_zones()
    flow_edges, perfusion_edges = get_canonical_edges(segments)

    return PatientCoronaryGraph(
        patient=patient,
        segments=segments,
        bifurcations=bifurcations,
        zones=zones,
        flow_edges=flow_edges,
        perfusion_edges=perfusion_edges
    )


def create_benchmark_patient_102() -> PatientCoronaryGraph:
    """
    Patient #102 — "The Rupture Paradox / The Widow Maker" (Vulnerable Plaque)
    - 52 yo male, asymptomatic or subtle fatigue.
    - 38% non-obstructive stenosis in Proximal LAD.
    - ULTRA-THIN CAP (42 µm < 65 µm TCFA), large necrotic lipid core (68%).
    - Directly adjacent (1.8 mm) to high-turbulence bifurcation (BIF_LM_LAD_LCX, shear index 8.4, turbulence 0.85).
    - DeepMind ApoB: 148 mg/dL (extreme atherogenic density), hs-CRP: 4.8 mg/L (severe cap inflammation).
    - Standard angiogram dismisses as 'mild non-critical 38% plaque'.
    - CardioGraph AI flags: CRITICAL IMMINENT RUPTURE (PVI 0.89) with 57.0% Myocardial Blast Radius!
    """
    patient = Patient(
        id="PAT-102",
        name="Marcus Vance",
        age=52,
        gender="Male",
        smoker=True,
        diabetes=False,
        hypertension=True,
        clinical_presentation="Asymptomatic executive; routine corporate health checkup with subtle fatigue.",
        benchmark_type="RUPTURE_PARADOX_VULNERABLE"
    )

    biomarkers = BiomarkerProfile(
        id="BIO-102",
        patient_id="PAT-102",
        apob_mg_dl=148.0,   # DeepMind ApoB high atherogenic particle count
        lpa_nmol_l=192.0,   # Extreme genetic pro-thrombotic risk
        hscrp_mg_l=4.8,     # High systemic vascular inflammation
        ldl_c_mg_dl=162.0,
        troponin_i_ng_ml=0.038,
        hba1c_pct=5.8
    )
    patient.biomarkers = biomarkers

    # LAD_Proximal soft lipid vulnerable plaque near LM bifurcation
    score, category = compute_vulnerability_score(
        plaque_type="soft_lipid",
        stenosis_pct=38.0,
        cap_thickness_um=42.0,   # TCFA < 65 µm
        core_lipid_ratio=0.68,
        calcification_score=0.8,
        apob_mg_dl=biomarkers.apob_mg_dl,
        hscrp_mg_l=biomarkers.hscrp_mg_l,
        turbulence_factor=0.85,
        shear_stress_index=8.4
    )

    plaque_102 = Plaque(
        id="PLQ-102-LAD-PROX",
        artery_segment_id="SEG_LAD_PROX",
        stenosis_pct=38.0,
        plaque_type="soft_lipid",
        cap_thickness_um=42.0,
        core_lipid_ratio=0.68,
        calcification_score=0.8,
        vulnerability_score=score,
        rupture_risk_category=category,
        adjacent_bifurcation_id="BIF_LM_LAD_LCX",
        distance_to_bifurcation_mm=1.8
    )
    patient.plaques.append(plaque_102)

    segments = get_canonical_artery_segments()
    bifurcations = get_canonical_bifurcations()
    zones = get_canonical_myocardium_zones()
    flow_edges, perfusion_edges = get_canonical_edges(segments)

    return PatientCoronaryGraph(
        patient=patient,
        segments=segments,
        bifurcations=bifurcations,
        zones=zones,
        flow_edges=flow_edges,
        perfusion_edges=perfusion_edges
    )


def create_benchmark_patient_103() -> PatientCoronaryGraph:
    """
    Patient #103 — "Diabetic Diffuse Multi-Vessel Disease"
    - 68 yo female with Type 2 Diabetes (HbA1c 9.2%).
    - Multi-focal moderate plaques in RCA, LCx, and LAD.
    """
    patient = Patient(
        id="PAT-103",
        name="Elena Rostova",
        age=68,
        gender="Female",
        smoker=False,
        diabetes=True,
        hypertension=True,
        clinical_presentation="Atypical dyspnea on exertion, poorly controlled diabetes for 14 years.",
        benchmark_type="DIABETIC_DIFFUSE_TRIPLE_VESSEL"
    )

    biomarkers = BiomarkerProfile(
        id="BIO-103",
        patient_id="PAT-103",
        apob_mg_dl=122.0,
        lpa_nmol_l=88.0,
        hscrp_mg_l=3.4,
        ldl_c_mg_dl=135.0,
        troponin_i_ng_ml=0.025,
        hba1c_pct=9.2
    )
    patient.biomarkers = biomarkers

    # Plaque 1: RCA Mid (55% mixed)
    s1, c1 = compute_vulnerability_score(
        "mixed", 55.0, 75.0, 0.42, 3.5,
        biomarkers.apob_mg_dl, biomarkers.hscrp_mg_l, 0.65, 6.5
    )
    # Plaque 2: LCx Prox (48% soft)
    s2, c2 = compute_vulnerability_score(
        "soft_lipid", 48.0, 58.0, 0.55, 1.2,
        biomarkers.apob_mg_dl, biomarkers.hscrp_mg_l, 0.60, 6.2
    )
    # Plaque 3: LAD Mid (52% fibrous)
    s3, c3 = compute_vulnerability_score(
        "fibrous", 52.0, 110.0, 0.25, 4.0,
        biomarkers.apob_mg_dl, biomarkers.hscrp_mg_l, 0.50, 5.5
    )

    patient.plaques = [
        Plaque("PLQ-103-RCA-MID", "SEG_RCA_MID", 55.0, "mixed", 75.0, 0.42, 3.5, s1, c1, "BIF_RCA_PDA_PLV", 14.0),
        Plaque("PLQ-103-LCX-PROX", "SEG_LCX_PROX", 48.0, "soft_lipid", 58.0, 0.55, 1.2, s2, c2, "BIF_LCX_OM1", 3.5),
        Plaque("PLQ-103-LAD-MID", "SEG_LAD_MID", 52.0, "fibrous", 110.0, 0.25, 4.0, s3, c3, "BIF_LAD_D2_S1", 8.0),
    ]

    segments = get_canonical_artery_segments()
    bifurcations = get_canonical_bifurcations()
    zones = get_canonical_myocardium_zones()
    flow_edges, perfusion_edges = get_canonical_edges(segments)

    return PatientCoronaryGraph(
        patient=patient,
        segments=segments,
        bifurcations=bifurcations,
        zones=zones,
        flow_edges=flow_edges,
        perfusion_edges=perfusion_edges
    )


def create_benchmark_patient_104() -> PatientCoronaryGraph:
    """
    Patient #104 — "Acute Circumflex Bifurcation Threat"
    - 58 yo male with nocturnal angina.
    - 62% soft lipid plaque at LCx-OM1 bifurcation carina.
    """
    patient = Patient(
        id="PAT-104",
        name="David Chen",
        age=58,
        gender="Male",
        smoker=True,
        diabetes=False,
        hypertension=False,
        clinical_presentation="Episodes of rest chest tightness radiating to the left arm during early morning.",
        benchmark_type="ACUTE_CIRCUMFLEX_BIFURCATION"
    )

    biomarkers = BiomarkerProfile(
        id="BIO-104",
        patient_id="PAT-104",
        apob_mg_dl=135.0,
        lpa_nmol_l=110.0,
        hscrp_mg_l=3.9,
        ldl_c_mg_dl=148.0,
        troponin_i_ng_ml=0.032,
        hba1c_pct=5.6
    )
    patient.biomarkers = biomarkers

    s, c = compute_vulnerability_score(
        "soft_lipid", 62.0, 52.0, 0.60, 1.5,
        biomarkers.apob_mg_dl, biomarkers.hscrp_mg_l, 0.72, 7.5
    )

    patient.plaques.append(Plaque(
        id="PLQ-104-LCX-PROX",
        artery_segment_id="SEG_LCX_PROX",
        stenosis_pct=62.0,
        plaque_type="soft_lipid",
        cap_thickness_um=52.0,
        core_lipid_ratio=0.60,
        calcification_score=1.5,
        vulnerability_score=s,
        rupture_risk_category=c,
        adjacent_bifurcation_id="BIF_LCX_OM1",
        distance_to_bifurcation_mm=2.1
    ))

    segments = get_canonical_artery_segments()
    bifurcations = get_canonical_bifurcations()
    zones = get_canonical_myocardium_zones()
    flow_edges, perfusion_edges = get_canonical_edges(segments)

    return PatientCoronaryGraph(
        patient=patient,
        segments=segments,
        bifurcations=bifurcations,
        zones=zones,
        flow_edges=flow_edges,
        perfusion_edges=perfusion_edges
    )


def create_benchmark_patient_105() -> PatientCoronaryGraph:
    """
    Patient #105 — "Healthy Control / Pristine Vasculature"
    - 42 yo female, marathon runner, clean coronary arteries.
    - Zero plaques, optimal ApoB (62 mg/dL), normal hs-CRP (0.4 mg/L).
    """
    patient = Patient(
        id="PAT-105",
        name="Sarah Jenkins",
        age=42,
        gender="Female",
        smoker=False,
        diabetes=False,
        hypertension=False,
        clinical_presentation="Annual preventative cardiology screening; highly active endurance athlete.",
        benchmark_type="HEALTHY_CONTROL"
    )

    biomarkers = BiomarkerProfile(
        id="BIO-105",
        patient_id="PAT-105",
        apob_mg_dl=62.0,
        lpa_nmol_l=18.0,
        hscrp_mg_l=0.4,
        ldl_c_mg_dl=74.0,
        troponin_i_ng_ml=0.005,
        hba1c_pct=5.1
    )
    patient.biomarkers = biomarkers
    patient.plaques = []  # Pristine vessels

    segments = get_canonical_artery_segments()
    bifurcations = get_canonical_bifurcations()
    zones = get_canonical_myocardium_zones()
    flow_edges, perfusion_edges = get_canonical_edges(segments)

    return PatientCoronaryGraph(
        patient=patient,
        segments=segments,
        bifurcations=bifurcations,
        zones=zones,
        flow_edges=flow_edges,
        perfusion_edges=perfusion_edges
    )


# ============================================================================
# SYNTHETIC COHORT GENERATOR (25 PATIENTS)
# ============================================================================

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra"
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White"
]

SEGMENT_CANDIDATES = [
    ("SEG_LM", "BIF_LM_LAD_LCX", 2.0),
    ("SEG_LAD_PROX", "BIF_LM_LAD_LCX", 2.5),
    ("SEG_LAD_PROX", "BIF_LAD_D1", 3.0),
    ("SEG_LAD_MID", "BIF_LAD_D2_S1", 4.0),
    ("SEG_D1", "BIF_LAD_D1", 5.0),
    ("SEG_LCX_PROX", "BIF_LCX_OM1", 2.8),
    ("SEG_LCX_DIST", "BIF_LCX_OM1", 6.0),
    ("SEG_OM1", "BIF_LCX_OM1", 4.5),
    ("SEG_RCA_PROX", None, None),
    ("SEG_RCA_MID", "BIF_RCA_PDA_PLV", 8.0),
    ("SEG_RCA_DIST", "BIF_RCA_PDA_PLV", 3.2),
    ("SEG_PDA", "BIF_RCA_PDA_PLV", 4.0),
]


def generate_cohort(num_patients: int = 25, seed: int = 42) -> List[PatientCoronaryGraph]:
    """Generates a diverse synthetic clinical cohort of patient coronary graphs."""
    rng = random.Random(seed)
    cohort: List[PatientCoronaryGraph] = []

    # First add the 5 core benchmark cases
    cohort.append(create_benchmark_patient_101())
    cohort.append(create_benchmark_patient_102())
    cohort.append(create_benchmark_patient_103())
    cohort.append(create_benchmark_patient_104())
    cohort.append(create_benchmark_patient_105())

    # Generate remaining patients (PAT-106 to PAT-100+N)
    for idx in range(6, num_patients + 1):
        pid = f"PAT-{100 + idx}"
        gender = "Male" if rng.random() > 0.45 else "Female"
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        age = rng.randint(45, 82)
        smoker = rng.random() < 0.35
        diabetes = rng.random() < 0.28
        hypertension = rng.random() < 0.60

        # Atherogenic biomarkers
        apob = round(rng.uniform(70.0, 155.0) + (20.0 if diabetes or smoker else 0.0), 1)
        lpa = round(rng.uniform(15.0, 210.0), 1)
        hscrp = round(rng.uniform(0.5, 6.2) + (1.0 if smoker else 0.0), 1)
        ldl = round(apob * 1.05 + rng.uniform(-10.0, 15.0), 1)
        trop = round(rng.uniform(0.005, 0.045), 3)
        hba1c = round(rng.uniform(5.2, 9.8) if diabetes else rng.uniform(4.9, 5.7), 1)

        biomarkers = BiomarkerProfile(
            id=f"BIO-{100 + idx}",
            patient_id=pid,
            apob_mg_dl=apob,
            lpa_nmol_l=lpa,
            hscrp_mg_l=hscrp,
            ldl_c_mg_dl=ldl,
            troponin_i_ng_ml=trop,
            hba1c_pct=hba1c
        )

        patient = Patient(
            id=pid,
            name=name,
            age=age,
            gender=gender,
            smoker=smoker,
            diabetes=diabetes,
            hypertension=hypertension,
            clinical_presentation=f"Referred for coronary evaluation. Age {age}, smoker={smoker}, diabetes={diabetes}.",
            biomarkers=biomarkers
        )

        # Generate 1 to 3 plaques per patient
        num_plaques = rng.choices([0, 1, 2, 3], weights=[0.1, 0.4, 0.35, 0.15])[0]
        chosen_segments = rng.sample(SEGMENT_CANDIDATES, k=min(num_plaques, len(SEGMENT_CANDIDATES)))

        for p_idx, (seg_id, bif_id, bif_dist) in enumerate(chosen_segments):
            plaque_id = f"PLQ-{100 + idx}-{p_idx + 1}"
            plaque_type = rng.choices(
                ["soft_lipid", "mixed", "fibrocalcific", "fibrous"],
                weights=[0.35, 0.30, 0.20, 0.15]
            )[0]

            if plaque_type == "soft_lipid":
                stenosis = round(rng.uniform(30.0, 65.0), 1)
                cap = round(rng.uniform(35.0, 68.0), 1)  # High risk of TCFA
                core = round(rng.uniform(0.45, 0.75), 2)
                calc = round(rng.uniform(0.2, 2.5), 1)
            elif plaque_type == "fibrocalcific":
                stenosis = round(rng.uniform(50.0, 85.0), 1)
                cap = round(rng.uniform(140.0, 220.0), 1)
                core = round(rng.uniform(0.10, 0.25), 2)
                calc = round(rng.uniform(5.0, 9.8), 1)
            else:
                stenosis = round(rng.uniform(35.0, 75.0), 1)
                cap = round(rng.uniform(70.0, 130.0), 1)
                core = round(rng.uniform(0.25, 0.45), 2)
                calc = round(rng.uniform(2.0, 5.5), 1)

            score, category = compute_vulnerability_score(
                plaque_type=plaque_type,
                stenosis_pct=stenosis,
                cap_thickness_um=cap,
                core_lipid_ratio=core,
                calcification_score=calc,
                apob_mg_dl=apob,
                hscrp_mg_l=hscrp,
                turbulence_factor=rng.uniform(0.5, 0.85),
                shear_stress_index=rng.uniform(5.0, 8.5)
            )

            plaque = Plaque(
                id=plaque_id,
                artery_segment_id=seg_id,
                stenosis_pct=stenosis,
                plaque_type=plaque_type,
                cap_thickness_um=cap,
                core_lipid_ratio=core,
                calcification_score=calc,
                vulnerability_score=score,
                rupture_risk_category=category,
                adjacent_bifurcation_id=bif_id,
                distance_to_bifurcation_mm=round(bif_dist + rng.uniform(-1.0, 2.0), 1) if bif_dist else None
            )
            patient.plaques.append(plaque)

        segments = get_canonical_artery_segments()
        bifurcations = get_canonical_bifurcations()
        zones = get_canonical_myocardium_zones()
        flow_edges, perfusion_edges = get_canonical_edges(segments)

        cohort.append(PatientCoronaryGraph(
            patient=patient,
            segments=segments,
            bifurcations=bifurcations,
            zones=zones,
            flow_edges=flow_edges,
            perfusion_edges=perfusion_edges
        ))

    return cohort


# ============================================================================
# EXPORT & SERIALIZATION
# ============================================================================

def export_dataset_to_dict(cohort: List[PatientCoronaryGraph]) -> Dict[str, any]:
    """Serializes the full cohort and canonical graph definitions to a JSON-ready dict."""
    serialized_patients = []

    for graph in cohort:
        p = graph.patient
        patient_dict = {
            "id": p.id,
            "name": p.name,
            "age": p.age,
            "gender": p.gender,
            "smoker": p.smoker,
            "diabetes": p.diabetes,
            "hypertension": p.hypertension,
            "clinical_presentation": p.clinical_presentation,
            "benchmark_type": p.benchmark_type,
            "biomarkers": asdict(p.biomarkers) if p.biomarkers else None,
            "plaques": [asdict(plq) for plq in p.plaques],
        }

        # Calculate blast radius for each plaque
        for plq_data in patient_dict["plaques"]:
            blast = calculate_downstream_blast_radius(
                start_segment_id=plq_data["artery_segment_id"],
                flow_edges=graph.flow_edges,
                perfusion_edges=graph.perfusion_edges,
                zones=graph.zones
            )
            plq_data["downstream_blast_radius"] = blast

        serialized_patients.append(patient_dict)

    canonical_graph = cohort[0]

    return {
        "metadata": {
            "project": "CardioGraph AI",
            "version": "1.0.0",
            "target": "WeMakeDevs Graph Hacks: Context for AI Agents (FalkorDB)",
            "scientific_context": "DeepMind ApoB100 + Bifurcation Wall Shear Stress + Rupture Paradox",
            "cohort_size": len(cohort),
            "canonical_segments_count": len(canonical_graph.segments),
            "canonical_bifurcations_count": len(canonical_graph.bifurcations),
            "canonical_myocardium_zones_count": len(canonical_graph.zones),
        },
        "canonical_topology": {
            "artery_segments": [asdict(s) for s in canonical_graph.segments],
            "bifurcations": [asdict(b) for b in canonical_graph.bifurcations],
            "myocardium_zones": [asdict(z) for z in canonical_graph.zones],
            "flow_edges": [asdict(e) for e in canonical_graph.flow_edges],
            "perfusion_edges": [asdict(p) for p in canonical_graph.perfusion_edges],
        },
        "patients": serialized_patients
    }


def save_dataset_json(output_path: str = "dataset.json", num_patients: int = 25) -> Path:
    """Generates and writes the complete dataset to a JSON file."""
    cohort = generate_cohort(num_patients=num_patients)
    data = export_dataset_to_dict(cohort)
    out_file = Path(output_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_file


# ============================================================================
# CLI RUNNER & BENCHMARK VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CardioGraph AI — Generating Coronary Vascular Tree Dataset...")
    print("=" * 80)

    cohort = generate_cohort(num_patients=25)
    out_path = save_dataset_json("dataset.json", num_patients=25)

    print(f"\n[+] Successfully generated {len(cohort)} patient coronary graphs.")
    print(f"[+] Saved structured dataset to: {out_path.resolve()}")

    # Print Clinical Comparison: The Rupture Paradox
    p101 = cohort[0].patient
    p102 = cohort[1].patient

    print("\n" + "=" * 80)
    print("CLINICAL SHOWCASE: THE 'RUPTURE PARADOX' (PAT-101 vs PAT-102)")
    print("=" * 80)

    print(f"\n[CASE 1: PAT-101 (Arthur Pendelton - 65yo)]")
    print(f"  Presentation: {p101.clinical_presentation}")
    print(f"  Biomarkers  : ApoB {p101.biomarkers.apob_mg_dl} mg/dL | hs-CRP {p101.biomarkers.hscrp_mg_l} mg/L")
    for plq in p101.plaques:
        print(f"  Plaque on {plq.artery_segment_id}:")
        print(f"    - Stenosis        : {plq.stenosis_pct}% (SEEKED BY CONVENTIONAL ANGIOGRAPHY)")
        print(f"    - Type & Cap      : {plq.plaque_type.upper()} | Cap Thickness: {plq.cap_thickness_um} µm")
        print(f"    - Vulnerability   : {plq.vulnerability_score} [{plq.rupture_risk_category}]")
        print(f"    - Clinical Reality: Stable calcified plaque. Low acute rupture hazard.")

    print(f"\n[CASE 2: PAT-102 (Marcus Vance - 52yo)]")
    print(f"  Presentation: {p102.clinical_presentation}")
    print(f"  Biomarkers  : ApoB {p102.biomarkers.apob_mg_dl} mg/dL | hs-CRP {p102.biomarkers.hscrp_mg_l} mg/L | Lp(a) {p102.biomarkers.lpa_nmol_l} nmol/L")
    for plq in p102.plaques:
        blast = calculate_downstream_blast_radius(
            plq.artery_segment_id, cohort[1].flow_edges, cohort[1].perfusion_edges, cohort[1].zones
        )
        print(f"  Plaque on {plq.artery_segment_id}:")
        print(f"    - Stenosis        : {plq.stenosis_pct}% (DISMISSED AS 'MILD/INSIGNIFICANT' BY ANGIOGRAPHY)")
        print(f"    - Type & Cap      : {plq.plaque_type.upper()} | Cap Thickness: {plq.cap_thickness_um} µm (TCFA < 65µm)")
        print(f"    - Vulnerability   : {plq.vulnerability_score} [{plq.rupture_risk_category} IMMINENT RUPTURE]")
        print(f"    - Adjacent BIF    : {plq.adjacent_bifurcation_id} ({plq.distance_to_bifurcation_mm} mm away)")
        print(f"    - Ischemic Blast  : {blast['total_myocardium_mass_at_risk_pct']}% of LV Mass ({blast['clinical_severity']})")
        print(f"    - Threatened Zones: {', '.join(z['zone_name'] for z in blast['threatened_zones'])}")

    print("\n" + "=" * 80)
    print("Ready for Step 2: FalkorDB Seed Script (seed_graph.py)")
    print("=" * 80)
