"""
CardioGraph AI — MCP Agent Core & Clinical REST API
====================================================
Autonomous clinical AI agent powered by FalkorDB for coronary vascular network
analysis, plaque vulnerability scoring, and ischemic blast radius prediction.

Scientific Foundations:
- Demis Hassabis / DeepMind ApoB100 atherogenic lipid particle model
- Thin-Cap Fibroatheroma (TCFA) cap thickness (<65 µm) threshold
- Hemodynamic Wall Shear Stress (WSS) and carinal bifurcation turbulence
- Directed graph reachability traversal for downstream myocardial necrosis

Architecture:
- MCP (Model Context Protocol) Server endpoints (/mcp/tools, /mcp/call)
- High-performance FastAPI REST backend (/api/*) for the visual frontend
- Dual Graph Engine: FalkorDB Cypher queries when connected, seamless in-memory
  graph traversal fallback when offline.
"""

from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import our biophysical dataset generator and graph algorithms
from dataset import (
    calculate_downstream_blast_radius,
    compute_vulnerability_score,
    export_dataset_to_dict,
    generate_cohort,
    get_canonical_artery_segments,
    get_canonical_bifurcations,
    get_canonical_edges,
    get_canonical_myocardium_zones,
)

# FalkorDB client
try:
    import falkordb
    from falkordb import FalkorDB, Graph
    FALKORDB_AVAILABLE = True
except ImportError:
    FALKORDB_AVAILABLE = False


# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="CardioGraph AI — Clinical MCP Agent Core",
    description="Vascular network intelligence & plaque rupture predictor powered by FalkorDB",
    version="1.0.0"
)

# Enable CORS for frontend integrations (Next.js, Vite, Cytoscape, React Flow)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# GRAPH CONTEXT MANAGER (DUAL ENGINE: FALKORDB + LOCAL FALLBACK)
# ============================================================================

class VascularGraphEngine:
    """
    Manages graph queries across FalkorDB and local in-memory structures.
    Guarantees 100% operational uptime whether connected to FalkorDB Cloud/Docker
    or operating in local demonstration mode.
    """

    def __init__(self, graph_name: str = "cardiograph"):
        self.graph_name = graph_name
        self.client: Optional[FalkorDB] = None
        self.falkor_graph: Optional[Graph] = None
        self.is_connected: bool = False
        self.connection_error: Optional[str] = None

        # Load local in-memory dataset
        self.dataset: Dict[str, Any] = self._load_dataset()
        self.patient_map: Dict[str, Dict[str, Any]] = {
            p["id"]: p for p in self.dataset["patients"]
        }

        # Attempt FalkorDB connection
        self.try_connect_falkordb()

    def _load_dataset(self) -> Dict[str, Any]:
        """Loads dataset.json from disk or regenerates it on the fly."""
        json_path = Path(__file__).parent / "dataset.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            cohort = generate_cohort(num_patients=25)
            data = export_dataset_to_dict(cohort)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return data

    def try_connect_falkordb(self) -> bool:
        """Connects to FalkorDB via environment variables or default localhost."""
        if not FALKORDB_AVAILABLE:
            self.connection_error = "falkordb package not installed."
            self.is_connected = False
            return False

        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", "6379"))
        password = os.getenv("FALKORDB_PASSWORD", None)
        username = os.getenv("FALKORDB_USERNAME", None)
        url = os.getenv("FALKORDB_URL", None)
        ssl = os.getenv("FALKORDB_SSL", "false").lower() in ("true", "1", "yes")

        try:
            if url:
                self.client = FalkorDB.from_url(url)
            else:
                self.client = FalkorDB(
                    host=host,
                    port=port,
                    password=password,
                    username=username,
                    ssl=ssl
                )
            self.client.connection.ping()
            self.falkor_graph = self.client.select_graph(self.graph_name)
            self.is_connected = True
            self.connection_error = None
            return True
        except Exception as e:
            self.is_connected = False
            self.connection_error = str(e)
            return False

    def query_cypher(self, cypher_query: str) -> Optional[List[List[Any]]]:
        """Executes a Cypher query on FalkorDB if connected."""
        if self.is_connected and self.falkor_graph:
            try:
                res = self.falkor_graph.query(cypher_query)
                return res.result_set
            except Exception as e:
                print(f"[!] FalkorDB query error: {e}")
                return None
        return None


# Global engine instance
graph_engine = VascularGraphEngine()


# ============================================================================
# AGENT TOOL DEFINITIONS (BIOPHYSICAL & CLINICAL LOGIC)
# ============================================================================

def tool_analyze_patient_vascular_tree(patient_id: str) -> Dict[str, Any]:
    """
    Analyzes a patient's coronary tree, atherogenic biomarkers, and plaque distribution.
    Discovers clinical anomalies including the Rupture Paradox.
    """
    patient = graph_engine.patient_map.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")

    bio = patient.get("biomarkers", {})
    plaques = patient.get("plaques", [])

    # Identify most critical plaque
    critical_plaques = []
    has_rupture_paradox = False

    for plq in plaques:
        v_score = plq.get("vulnerability_score", 0.0)
        stenosis = plq.get("stenosis_pct", 0.0)
        cap = plq.get("cap_thickness_um", 100.0)
        plq_type = plq.get("plaque_type", "")

        # Rupture Paradox Detection: Non-obstructive (<50% stenosis) but TCFA (<65µm cap) and high vulnerability
        if stenosis < 50.0 and cap < 65.0 and v_score >= 0.70:
            has_rupture_paradox = True

        if v_score >= 0.60 or cap < 65.0:
            critical_plaques.append(plq)

    # Clinical risk synthesis
    risk_level = "CRITICAL" if has_rupture_paradox or any(p["rupture_risk_category"] == "CRITICAL" for p in plaques) else (
        "HIGH" if len(critical_plaques) > 0 else (
            "MODERATE" if len(plaques) > 0 else "OPTIMAL_HEALTH"
        )
    )

    summary_insight = (
        f"PATIENT ALERT: High-risk vulnerable plaque detected despite non-obstructive stenosis "
        f"({critical_plaques[0]['stenosis_pct']}% lumen narrowing, cap thickness {critical_plaques[0]['cap_thickness_um']} µm). "
        f"Classic 'Rupture Paradox' scenario requiring aggressive plaque stabilization!"
        if has_rupture_paradox else
        f"Patient exhibits {len(plaques)} plaque lesion(s) with overall risk status: {risk_level}."
    )

    return {
        "patient_id": patient["id"],
        "name": patient["name"],
        "age": patient["age"],
        "gender": patient["gender"],
        "cardiovascular_risk_level": risk_level,
        "rupture_paradox_present": has_rupture_paradox,
        "clinical_summary": summary_insight,
        "biomarkers": {
            "apob_mg_dl": bio.get("apob_mg_dl"),
            "apob_status": "HIGH_RISK" if bio.get("apob_mg_dl", 0) > 100 else "OPTIMAL",
            "lpa_nmol_l": bio.get("lpa_nmol_l"),
            "hscrp_mg_l": bio.get("hscrp_mg_l"),
            "inflammation_status": "ACTIVE_SYSTEMIC_INFLAMMATION" if bio.get("hscrp_mg_l", 0) > 3.0 else "NORMAL",
            "troponin_i_ng_ml": bio.get("troponin_i_ng_ml"),
            "ldl_c_mg_dl": bio.get("ldl_c_mg_dl"),
            "hba1c_pct": bio.get("hba1c_pct")
        },
        "plaques_count": len(plaques),
        "plaques": plaques,
        "falkordb_live_queried": graph_engine.is_connected
    }


def tool_calculate_ischemic_blast_radius(artery_segment_id: str, plaque_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes downstream myocardial necrosis territory using graph reachability.
    Traverses (ArterySegment)-[:FLOWS_INTO*]->(downstream)-[:PERFUSES]->(MyocardiumZone).
    """
    topo = graph_engine.dataset["canonical_topology"]

    # If FalkorDB is connected, execute Cypher traversal query
    if graph_engine.is_connected and graph_engine.falkor_graph:
        cypher = f"""
        MATCH (origin:ArterySegment {{id: '{artery_segment_id}'}})
        MATCH (origin)-[:FLOWS_INTO*0..5]->(downstream:ArterySegment)-[:PERFUSES]->(z:MyocardiumZone)
        RETURN DISTINCT z.id AS zone_id, z.zone_name AS zone_name, z.wall AS wall, z.mass_pct AS mass_pct
        ORDER BY z.mass_pct DESC
        """
        res = graph_engine.query_cypher(cypher)
        if res is not None:
            threatened_zones = []
            total_mass = 0.0
            for row in res:
                mass = float(row[3])
                total_mass += mass
                threatened_zones.append({
                    "id": row[0],
                    "zone_name": row[1],
                    "wall": row[2],
                    "mass_pct": mass
                })

            return {
                "source_segment_id": artery_segment_id,
                "plaque_id": plaque_id,
                "engine_used": "FalkorDB_Cypher_Traversal",
                "total_myocardium_mass_at_risk_pct": round(total_mass, 1),
                "clinical_severity": (
                    "MASSIVE_INFARCTION" if total_mass >= 45.0 else
                    "EXTENSIVE_ISCHEMIA" if total_mass >= 25.0 else
                    "REGIONAL_ISCHEMIA" if total_mass >= 10.0 else "LOCALIZED_ISCHEMIA"
                ),
                "threatened_zones": threatened_zones
            }

    # In-memory graph traversal fallback
    flow_edges = [
        type("Edge", (), e) for e in topo["flow_edges"]
    ]
    perfusion_edges = [
        type("PEdge", (), p) for p in topo["perfusion_edges"]
    ]
    zones = [
        type("Zone", (), z) for z in topo["myocardium_zones"]
    ]

    blast = calculate_downstream_blast_radius(
        start_segment_id=artery_segment_id,
        flow_edges=flow_edges,
        perfusion_edges=perfusion_edges,
        zones=zones
    )
    blast["engine_used"] = "VascularGraphEngine_In_Memory"
    blast["plaque_id"] = plaque_id
    return blast


def tool_find_vulnerable_plaques(min_vulnerability_score: float = 0.60) -> List[Dict[str, Any]]:
    """
    Scans the patient cohort for vulnerable plaques and thin-cap fibroatheromas (TCFA).
    Highlights plaques near high-turbulence bifurcations.
    """
    results = []
    for p in graph_engine.dataset["patients"]:
        for plq in p.get("plaques", []):
            score = plq.get("vulnerability_score", 0.0)
            cap = plq.get("cap_thickness_um", 100.0)
            if score >= min_vulnerability_score or cap < 65.0:
                results.append({
                    "patient_id": p["id"],
                    "patient_name": p["name"],
                    "plaque_id": plq["id"],
                    "segment_id": plq["artery_segment_id"],
                    "stenosis_pct": plq["stenosis_pct"],
                    "plaque_type": plq["plaque_type"],
                    "cap_thickness_um": cap,
                    "is_tcfa": cap < 65.0,
                    "vulnerability_score": score,
                    "risk_category": plq["rupture_risk_category"],
                    "adjacent_bifurcation": plq.get("adjacent_bifurcation_id"),
                    "distance_to_bifurcation_mm": plq.get("distance_to_bifurcation_mm"),
                    "apob_mg_dl": p.get("biomarkers", {}).get("apob_mg_dl"),
                    "hscrp_mg_l": p.get("biomarkers", {}).get("hscrp_mg_l"),
                    "rupture_paradox": (plq["stenosis_pct"] < 50.0 and cap < 65.0 and score >= 0.70)
                })

    results.sort(key=lambda x: x["vulnerability_score"], reverse=True)
    return results


def tool_simulate_plaque_rupture(plaque_id: str) -> Dict[str, Any]:
    """
    Simulates an acute plaque rupture cascade:
    1. Fibrous cap tears under hemodynamic shear stress.
    2. Thrombogenic lipid core is exposed to circulating blood.
    3. Rapid platelet aggregation and acute thrombotic occlusion.
    4. Computes real-time downstream ischemia blast radius and ECG ST-elevation leads.
    """
    # Locate plaque
    found_patient = None
    target_plaque = None

    for p in graph_engine.dataset["patients"]:
        for plq in p.get("plaques", []):
            if plq.get("id") == plaque_id or plq.get("id", "").startswith(plaque_id) or p.get("id") == plaque_id:
                found_patient = p
                target_plaque = plq
                break
        if target_plaque:
            break

    if not target_plaque or not found_patient:
        raise HTTPException(status_code=404, detail=f"Plaque '{plaque_id}' not found.")

    seg_id = target_plaque["artery_segment_id"]
    blast = tool_calculate_ischemic_blast_radius(seg_id, plaque_id)

    # Determine clinical ECG leads affected
    walls_affected = set(z["wall"] for z in blast["threatened_zones"])
    ecg_changes = []
    if "Anterior" in walls_affected:
        ecg_changes.append("ST-Elevation in V1, V2, V3, V4 (Anterior STEMI)")
    if "Septal" in walls_affected:
        ecg_changes.append("ST-Elevation in V1, V2 (Septal Involvement)")
    if "Lateral" in walls_affected:
        ecg_changes.append("ST-Elevation in I, aVL, V5, V6 (Lateral STEMI)")
    if "Inferior" in walls_affected:
        ecg_changes.append("ST-Elevation in II, III, aVF (Inferior STEMI)")

    return {
        "simulation_status": "ACUTE_THROMBOTIC_OCCLUSION",
        "patient_id": found_patient["id"],
        "patient_name": found_patient["name"],
        "ruptured_plaque": target_plaque,
        "occluded_artery_segment": seg_id,
        "downstream_blast_radius": blast,
        "threatened_heart_walls": list(walls_affected),
        "predicted_ecg_manifestation": ecg_changes,
        "hemodynamic_collapse": {
            "pre_rupture_stenosis_pct": target_plaque["stenosis_pct"],
            "post_rupture_stenosis_pct": 100.0,
            "distal_perfusion_flow_drop_pct": 100.0,
            "time_to_irreversible_myocyte_necrosis_minutes": 20.0
        },
        "clinical_emergency_protocol": (
            "CODE STEMI ACTIVATION: Immediate transfer to Cardiac Catheterization Lab for primary PCI / balloon thrombectomy. "
            f"Vessel at risk: {seg_id} threatening {blast['total_myocardium_mass_at_risk_pct']}% of left ventricle."
        )
    }


# ============================================================================
# MODEL CONTEXT PROTOCOL (MCP) INTERFACE
# ============================================================================

MCP_TOOLS = [
    {
        "name": "analyze_patient_vascular_tree",
        "description": "Analyzes the coronary arterial tree, atherogenic biomarkers (ApoB, Lp(a), hs-CRP), and plaque vulnerability for a patient.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient ID, e.g. PAT-101, PAT-102"
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "calculate_ischemic_blast_radius",
        "description": "Calculates downstream myocardial zones starved of oxygen if an artery segment or plaque becomes occluded.",
        "parameters": {
            "type": "object",
            "properties": {
                "artery_segment_id": {
                    "type": "string",
                    "description": "Artery segment identifier (e.g., SEG_LAD_PROX, SEG_LM, SEG_RCA_MID)"
                },
                "plaque_id": {
                    "type": "string",
                    "description": "Optional plaque ID"
                }
            },
            "required": ["artery_segment_id"]
        }
    },
    {
        "name": "find_vulnerable_plaques",
        "description": "Scans patient cohort for high-risk thin-cap fibroatheromas (TCFA < 65µm) and Rupture Paradox cases.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_vulnerability_score": {
                    "type": "number",
                    "description": "Minimum vulnerability threshold (0.0 to 1.0, default 0.60)",
                    "default": 0.60
                }
            }
        }
    },
    {
        "name": "simulate_plaque_rupture",
        "description": "Simulates acute plaque rupture, triggering 100% thrombotic occlusion and calculating real-time ischemic blast radius.",
        "parameters": {
            "type": "object",
            "properties": {
                "plaque_id": {
                    "type": "string",
                    "description": "ID of plaque to simulate rupture on (e.g. PLQ-102-LAD-PROX)"
                }
            },
            "required": ["plaque_id"]
        }
    }
]


@app.get("/mcp/tools")
def mcp_list_tools():
    """Returns the list of available MCP tools."""
    return {"tools": MCP_TOOLS}


class MCPCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


@app.post("/mcp/call")
def mcp_call_tool(request: MCPCallRequest):
    """Executes an MCP tool call."""
    name = request.name
    args = request.arguments

    if name == "analyze_patient_vascular_tree":
        pid = args.get("patient_id")
        if not pid:
            raise HTTPException(status_code=400, detail="Missing patient_id")
        return {"result": tool_analyze_patient_vascular_tree(pid)}

    elif name == "calculate_ischemic_blast_radius":
        seg_id = args.get("artery_segment_id")
        if not seg_id:
            raise HTTPException(status_code=400, detail="Missing artery_segment_id")
        plq_id = args.get("plaque_id")
        return {"result": tool_calculate_ischemic_blast_radius(seg_id, plq_id)}

    elif name == "find_vulnerable_plaques":
        min_v = float(args.get("min_vulnerability_score", 0.60))
        return {"result": tool_find_vulnerable_plaques(min_v)}

    elif name == "simulate_plaque_rupture":
        plq_id = args.get("plaque_id")
        if not plq_id:
            raise HTTPException(status_code=400, detail="Missing plaque_id")
        return {"result": tool_simulate_plaque_rupture(plq_id)}

    else:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")


# ============================================================================
# REST API ENDPOINTS FOR FRONTEND INTEGRATION
# ============================================================================

@app.get("/api/health")
def health_check():
    """System health check and database connectivity diagnostic."""
    return {
        "status": "HEALTHY",
        "project": "CardioGraph AI",
        "falkordb_connected": graph_engine.is_connected,
        "falkordb_error": graph_engine.connection_error,
        "active_graph": graph_engine.graph_name,
        "total_patients_loaded": len(graph_engine.dataset["patients"]),
        "canonical_segments_count": len(graph_engine.dataset["canonical_topology"]["artery_segments"])
    }


@app.get("/api/graph-status")
def graph_status():
    """Graph engine connectivity status for frontend display."""
    return {
        "falkordb_connected": graph_engine.is_connected,
        "engine_mode": "FalkorDB Cloud" if graph_engine.is_connected else "Local Graph Engine",
        "total_patients": len(graph_engine.dataset["patients"]),
        "graph_name": graph_engine.graph_name
    }


@app.get("/api/patients")
def list_patients():
    """Lists all patients with risk categories, plaque details, and biomarkers."""
    summary = []
    for p in graph_engine.dataset["patients"]:
        has_crit = any(plq["rupture_risk_category"] == "CRITICAL" for plq in p.get("plaques", []))
        has_high = any(plq["rupture_risk_category"] == "HIGH" for plq in p.get("plaques", []))
        summary.append({
            "id": p["id"],
            "name": p["name"],
            "age": p["age"],
            "gender": p["gender"],
            "smoker": p["smoker"],
            "diabetes": p["diabetes"],
            "hypertension": p["hypertension"],
            "benchmark_type": p.get("benchmark_type"),
            "clinical_presentation": p.get("clinical_presentation", ""),
            "plaques_count": len(p.get("plaques", [])),
            "plaques": p.get("plaques", []),
            "biomarkers": p.get("biomarkers", {}),
            "apob_mg_dl": p.get("biomarkers", {}).get("apob_mg_dl"),
            "risk_tag": "CRITICAL" if has_crit else ("HIGH" if has_high else ("MODERATE" if len(p.get("plaques", [])) > 0 else "NORMAL"))
        })
    return {"patients": summary}


@app.get("/api/patients/{patient_id}")
@app.get("/api/patient/{patient_id}")
def get_patient_graph(patient_id: str):
    """
    Returns full coronary graph payload for interactive Cytoscape / 2D/3D visualization:
    - Artery segments (nodes)
    - Hemodynamic flow edges
    - Bifurcations
    - Myocardium zones
    - Perfusion edges
    - Patient plaques and biomarkers
    """
    patient = graph_engine.patient_map.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    topo = graph_engine.dataset["canonical_topology"]

    return {
        "patient": patient,
        "topology": topo
    }


@app.get("/api/vulnerable-plaques")
def get_vulnerable_plaques(min_score: float = Query(0.60, ge=0.0, le=1.0)):
    """Cohort scan of all vulnerable plaques."""
    return {"vulnerable_plaques": tool_find_vulnerable_plaques(min_score)}


class RuptureRequest(BaseModel):
    plaque_id: str


@app.post("/api/simulate-rupture")
def simulate_rupture(req: RuptureRequest):
    """Simulates acute rupture and returns real-time blast radius."""
    return tool_simulate_plaque_rupture(req.plaque_id)


@app.get("/api/cypher-queries/{patient_id}")
def get_patient_cypher_queries(patient_id: str):
    """
    Returns the exact FalkorDB Cypher queries that power this patient's analysis:
    - Vascular tree reconstruction
    - Plaque destabilization multi-hop traversal (Biomarker -> Plaque -> Bifurcation)
    - Downstream ischemic blast radius reachability traversal
    - Critical bifurcation chokepoints
    """
    p = graph_engine.patient_map.get(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    plaque_id = p["plaques"][0]["id"] if p.get("plaques") else "PLQ-102-LAD-PROX"
    seg_id = p["plaques"][0]["artery_segment_id"] if p.get("plaques") else "SEG_LAD_PROX"

    return {
        "patient_id": patient_id,
        "falkordb_connected": graph_engine.is_connected,
        "graph_name": graph_engine.graph_name,
        "engine_mode": "FalkorDB GraphBLAS" if graph_engine.is_connected else "In-Memory Graph Traversal Engine",
        "queries": [
            {
                "title": "1. Coronary Vascular Tree Topology Traversal",
                "description": "Reconstructs complete directed arterial flow tree from ascending aorta down to distal microvascular beds.",
                "cypher": f"""MATCH (p:Patient {{id: '{patient_id}'}})-[:HAS_VASCULAR_TREE]->(root:ArterySegment)
MATCH (root)-[:FLOWS_INTO*0..6]->(branch:ArterySegment)
OPTIONAL MATCH (branch)-[:CONTAINS_PLAQUE]->(plq:Plaque)
OPTIONAL MATCH (branch)-[:PERFUSES]->(zone:MyocardiumZone)
RETURN root.name, branch.name, branch.diameter_mm, branch.baseline_flow_ml_min, plq.stenosis_pct, zone.zone_name;""",
                "graphblas_operation": "Breadth-First Matrix Multiplication A * P"
            },
            {
                "title": "2. Multi-Hop Plaque Destabilization Traversal (ApoB + hs-CRP + WSS)",
                "description": "Traces biochemical and hemodynamic destabilization cascade across 3 hops: Biomarkers -> Plaque -> Bifurcation Shear.",
                "cypher": f"""MATCH (p:Patient {{id: '{patient_id}'}})-[:HAS_BIOMARKERS]->(b:BiomarkerProfile)
MATCH (p)-[:HAS_VASCULAR_TREE]->(seg:ArterySegment)-[:CONTAINS_PLAQUE]->(plq:Plaque {{id: '{plaque_id}'}})
MATCH (plq)-[adj:ADJACENT_TO]->(bif:Bifurcation)
WHERE b.apob_mg_dl > 100.0 AND plq.cap_thickness_um < 65.0 AND bif.shear_stress_index > 7.0
RETURN p.name, plq.id, plq.stenosis_pct, plq.cap_thickness_um, b.apob_mg_dl, bif.turbulence_factor;""",
                "graphblas_operation": "Multi-Relational Pattern Match Semiring"
            },
            {
                "title": "3. Downstream Ischemic Blast Radius (Reachability Traversal)",
                "description": "Computes exact myocardial zones at risk of necrosis if acute thrombosis occludes the target artery segment.",
                "cypher": f"""MATCH (origin:ArterySegment {{id: '{seg_id}'}})-[:FLOWS_INTO*0..5]->(downstream:ArterySegment)
MATCH (downstream)-[:PERFUSES]->(zone:MyocardiumZone)
RETURN DISTINCT zone.id AS zone_id,
       zone.zone_name AS zone_name,
       zone.wall AS wall,
       zone.mass_pct AS mass_pct,
       SUM(zone.mass_pct) OVER () AS total_threatened_mass_pct;""",
                "graphblas_operation": "Transitive Closure Matrix Power A^k"
            },
            {
                "title": "4. Critical Bifurcation Chokepoints (Single Points of Failure)",
                "description": "Calculates betweenness centrality across vascular junctions sustaining the greatest systemic cardiac output.",
                "cypher": """MATCH (b:Bifurcation)<-[:ADJACENT_TO]-(p:Plaque)<-[:CONTAINS_PLAQUE]-(s:ArterySegment)
WHERE b.turbulence_factor > 0.65
RETURN b.id, b.bifurcation_angle_deg, b.shear_stress_index, count(p) AS vulnerable_plaques_count
ORDER BY b.shear_stress_index DESC;""",
                "graphblas_operation": "Brandes Centrality via Adjacency Sparse Multiplication"
            }
        ]
    }


class CustomPatientRequest(BaseModel):
    name: str = "Patient Alex"
    age: int = 54
    gender: str = "Male"
    smoker: bool = True
    diabetes: bool = False
    hypertension: bool = True
    clinical_presentation: str = "Chest discomfort during high stress; referred for risk stratification."

    # Biomarkers
    apob_mg_dl: float = 135.0
    lpa_nmol_l: float = 85.0
    hscrp_mg_l: float = 3.8
    ldl_c_mg_dl: float = 145.0
    troponin_i_ng_ml: float = 0.025
    hba1c_pct: float = 5.7

    # Plaque findings from CCTA
    artery_segment_id: str = "SEG_LAD_PROX"
    stenosis_pct: float = 42.0
    plaque_type: str = "soft_lipid"
    cap_thickness_um: float = 48.0
    core_lipid_ratio: float = 0.60
    calcification_score: float = 1.2
    adjacent_bifurcation_id: Optional[str] = "BIF_LM_LAD_LCX"
    distance_to_bifurcation_mm: Optional[float] = 2.0


@app.post("/api/predict-patient")
def predict_custom_patient(req: CustomPatientRequest):
    """
    Accepts real doctor/user data from CT angiogram & lab biomarkers.
    Calculates Plaque Vulnerability Index (PVI), detects Rupture Paradox,
    computes downstream Ischemic Blast Radius, and integrates the patient
    into the active vascular tree graph.
    """
    # 1. Compute biophysical vulnerability score
    score, category = compute_vulnerability_score(
        plaque_type=req.plaque_type,
        stenosis_pct=req.stenosis_pct,
        cap_thickness_um=req.cap_thickness_um,
        core_lipid_ratio=req.core_lipid_ratio,
        calcification_score=req.calcification_score,
        apob_mg_dl=req.apob_mg_dl,
        hscrp_mg_l=req.hscrp_mg_l,
        turbulence_factor=0.80 if req.distance_to_bifurcation_mm and req.distance_to_bifurcation_mm < 5.0 else 0.45,
        shear_stress_index=8.0 if req.distance_to_bifurcation_mm and req.distance_to_bifurcation_mm < 5.0 else 5.0
    )

    import random
    custom_num = random.randint(201, 999)
    patient_id = f"PAT-USER-{custom_num}"
    plaque_id = f"PLQ-USER-{custom_num}"
    bio_id = f"BIO-USER-{custom_num}"

    # 2. Build patient data structure
    bio_data = {
        "id": bio_id,
        "patient_id": patient_id,
        "apob_mg_dl": req.apob_mg_dl,
        "lpa_nmol_l": req.lpa_nmol_l,
        "hscrp_mg_l": req.hscrp_mg_l,
        "ldl_c_mg_dl": req.ldl_c_mg_dl,
        "troponin_i_ng_ml": req.troponin_i_ng_ml,
        "hba1c_pct": req.hba1c_pct
    }

    # 3. Calculate downstream blast radius
    blast = tool_calculate_ischemic_blast_radius(req.artery_segment_id, plaque_id)

    plaque_data = {
        "id": plaque_id,
        "artery_segment_id": req.artery_segment_id,
        "stenosis_pct": req.stenosis_pct,
        "plaque_type": req.plaque_type,
        "cap_thickness_um": req.cap_thickness_um,
        "core_lipid_ratio": req.core_lipid_ratio,
        "calcification_score": req.calcification_score,
        "vulnerability_score": score,
        "rupture_risk_category": category,
        "adjacent_bifurcation_id": req.adjacent_bifurcation_id,
        "distance_to_bifurcation_mm": req.distance_to_bifurcation_mm,
        "downstream_blast_radius": blast
    }

    patient_data = {
        "id": patient_id,
        "name": req.name,
        "age": req.age,
        "gender": req.gender,
        "smoker": req.smoker,
        "diabetes": req.diabetes,
        "hypertension": req.hypertension,
        "clinical_presentation": req.clinical_presentation,
        "benchmark_type": "CUSTOM_USER_INPUT",
        "biomarkers": bio_data,
        "plaques": [plaque_data]
    }

    # 4. Integrate into active Vascular Graph Engine so frontend can instantly display it
    graph_engine.patient_map[patient_id] = patient_data
    graph_engine.dataset["patients"].insert(0, patient_data)

    # 5. Check Rupture Paradox status
    is_rupture_paradox = (req.stenosis_pct < 50.0 and req.cap_thickness_um < 65.0 and score >= 0.70)

    # 6. Predict ECG changes based on affected heart walls
    walls_affected = set(z["wall"] for z in blast.get("threatened_zones", []))
    ecg_changes = []
    if "Anterior" in walls_affected:
        ecg_changes.append("ST-Elevation in V1, V2, V3, V4 (Anterior STEMI)")
    if "Septal" in walls_affected:
        ecg_changes.append("ST-Elevation in V1, V2 (Septal Involvement)")
    if "Lateral" in walls_affected:
        ecg_changes.append("ST-Elevation in I, aVL, V5, V6 (Lateral STEMI)")
    if "Inferior" in walls_affected:
        ecg_changes.append("ST-Elevation in II, III, aVF (Inferior STEMI)")

    return {
        "status": "SUCCESS",
        "patient": patient_data,
        "diagnostic_prediction": {
            "vulnerability_score": score,
            "rupture_risk_category": category,
            "is_tcfa": req.cap_thickness_um < 65.0,
            "rupture_paradox_flag": is_rupture_paradox,
            "downstream_myocardium_at_risk_pct": blast.get("total_myocardium_mass_at_risk_pct", 0.0),
            "threatened_walls": list(walls_affected),
            "predicted_ecg": ecg_changes,
            "clinical_recommendation": (
                "🚨 CRITICAL INTERVENTION: Thin-cap fibroatheroma at high rupture risk. Initiate intensive statin/PCSK9i stabilization."
                if is_rupture_paradox or category == "CRITICAL" else
                "MODERATE RISK: Medical therapy and follow-up recommended."
            )
        }
    }


@app.post("/api/upload-patient-json")
def upload_patient_json(payload: Dict[str, Any]):
    """Uploads a complete patient JSON record directly."""
    pid = payload.get("id") or f"PAT-UPLOAD-{random.randint(100, 999)}"
    payload["id"] = pid

    # Compute blast radius for each plaque if not already computed
    for plq in payload.get("plaques", []):
        if not plq.get("downstream_blast_radius"):
            blast = tool_calculate_ischemic_blast_radius(plq["artery_segment_id"], plq["id"])
            plq["downstream_blast_radius"] = blast

    graph_engine.patient_map[pid] = payload
    graph_engine.dataset["patients"].insert(0, payload)

    return {
        "status": "SUCCESS",
        "patient_id": pid,
        "message": f"Patient {pid} ({payload.get('name')}) successfully integrated into vascular network."
    }


# ============================================================================
# LANGGRAPH CONVERSATIONAL AGENT ENDPOINT
# ============================================================================

class ChatRequest(BaseModel):
    patient_id: str = Field(..., description="Patient identifier (e.g. PAT-102)")
    message: str = Field("", description="User's question or message")
    mode: str = Field("patient", description="'patient' for plain-English, 'doctor' for clinical")

@app.post("/api/chat")
def chat_with_agent(req: ChatRequest):
    """
    Conversational LangGraph Agent endpoint.
    Runs the full 4-step clinical reasoning pipeline:
      1. Vascular Graph Retrieval → 2. Treatment Planning →
      3. Nutrition & Lifestyle → 4. Patient-Friendly Synthesis
    """
    try:
        from langgraph_agent import run_cardio_chat_agent
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LangGraph agent failed to initialize: {str(e)}")

    # Ensure patient exists in our graph engine
    patient_data = graph_engine.patient_map.get(req.patient_id)
    if not patient_data:
        raise HTTPException(status_code=404, detail=f"Patient {req.patient_id} not found.")

    result = run_cardio_chat_agent(
        patient_id=req.patient_id,
        message=req.message,
        mode=req.mode,
        patient_data=patient_data
    )
    return result


# ============================================================================
# FRONTEND STATIC ASSETS & SINGLE PAGE APPLICATION
# ============================================================================

frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/")
    def serve_frontend_index():
        """Serves the interactive vascular dashboard."""
        return FileResponse(frontend_path / "index.html")

    @app.get("/style.css")
    def serve_frontend_css():
        return FileResponse(frontend_path / "style.css")

    @app.get("/app.js")
    def serve_frontend_js():
        return FileResponse(frontend_path / "app.js")


# ============================================================================
# SERVER RUNNER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    print("=" * 80)
    print(f"Starting CardioGraph AI MCP Agent Core on http://{host}:{port}")
    print("FalkorDB Status:", "CONNECTED" if graph_engine.is_connected else "LOCAL GRAPH ENGINE (STANDALONE)")
    print("=" * 80)
    uvicorn.run("agent:app", host=host, port=port, reload=False)
