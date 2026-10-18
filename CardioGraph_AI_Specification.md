# CardioGraph AI — Master Clinical & Technical Blueprint
### Autonomous Coronary Vascular Network & Plaque Rupture Predictor
**Hackathon Target**: WeMakeDevs "Graph Hacks: Context for AI Agents" (Powered by FalkorDB)  
**Scientific Alignment**: Sir Demis Hassabis (DeepMind / Isomorphic Labs) & Dario Amodei (Anthropic)  
**Project Location**: `d:\MIT\CardioGraph-AI` and `D:\CardioGraph AI`

---

## 1. Executive Summary

Cardiovascular disease (CVD) is the **#1 killer on Earth**, claiming 18 million lives every year. In late 2025, Nobel Laureate **Sir Demis Hassabis** and Google DeepMind utilized AlphaFold to map the molecular structure of **ApoB100**—the foundational protein scaffold of LDL ("bad cholesterol"). Simultaneously, **Dario Amodei** (CEO of Anthropic, Ph.D. in Biophysics) identified the systemic causal modeling of cardiovascular disease as one of the most critical frontiers for artificial intelligence.

While AlphaFold resolved the static molecular shape of cholesterol proteins, modern medicine remains blind to the systemic dynamic: **how plaque builds up, destabilizes at turbulent arterial branching junctions, and causes sudden fatal heart attacks**.

**CardioGraph AI** bridges this gap. It is an autonomous clinical AI agent powered by **FalkorDB** as its native context layer. It reconstructs a patient's coronary arterial tree as a directed graph, calculates hemodynamic shear turbulence at arterial bifurcations, traces multi-hop biochemical destabilization (ApoB + hs-CRP), and computes the **exact ischemic blast radius** (heart muscle zones at risk of necrosis) before a fatal rupture occurs.

---

## 2. Who Will Use It & Who Pays For It?

| User / Stakeholder | Clinical Setting | Daily Unsolved Pain Point |
|---|---|---|
| **Interventional Cardiologists** | Cardiac Catheterization Lab | Deciding whether a 30–50% non-obstructive plaque requires aggressive intervention or stent, or if it is at high risk of sudden rupture. |
| **Cardiac Radiologists** | CCTA Imaging Department | Tracing 15+ arterial branches across 500+ CT slices manually takes 45 minutes per patient; they need an instant vascular network graph. |
| **Emergency Room Physicians** | ER Chest Pain Triage | Differentiating benign chest pain from an impending massive myocardial infarction based on branch perfusion risk. |
| **Hospitals & Insurers (Buyers)** | Health Systems / Payers | A single emergency ICU heart attack costs \$50,000–\$150,000. Early graph-based plaque stabilization saves millions per network. |

---

## 3. The Clinical Problem: "The Rupture Paradox"

- **The Standard Blind Spot**: In current clinical practice, angiograms highlight high-stenosis blockages (e.g., 70% or 80%). Doctors stent these "pipes."
- **The Deadly Reality**: **Over 65% of sudden fatal heart attacks occur at plaques with LESS than 50% stenosis.**
- **Why?** These are "soft, lipid-rich vulnerable plaques" with thin fibrous caps. When they sit near high-turbulence arterial forks (bifurcations), hemodynamic shear stress tears the cap, triggering sudden thrombosis (the "Widow Maker" heart attack).
- **Why Graph Database is Required**: An arterial network is physically and mathematically a **directed flow graph**. SQL tables cannot compute recursive downstream perfusion, and flat vector embeddings cannot model fluid hemodynamics.

---

## 4. FalkorDB Vascular Graph Schema

### Node Labels
1. `Patient`: `{id, age, gender, smoker, diabetes}`
2. `ArterySegment`: `{name, diameter_mm, length_mm, baseline_flow_ml_min}`
   - Examples: `Left_Main`, `LAD_Proximal`, `LAD_Mid`, `LAD_Distal`, `LCx`, `RCA`, `Diagonal_1`, `OM_1`.
3. `Bifurcation`: `{id, bifurcation_angle_deg, shear_stress_index, turbulence_factor}`
4. `Plaque`: `{id, stenosis_pct, plaque_type: 'soft_lipid' | 'fibrocalcific', cap_thickness_um, vulnerability_score}`
5. `BiomarkerProfile`: `{apob_mg_dl, lpa_nmol_l, hscrp_mg_l, troponin_i}`
6. `MyocardiumZone`: `{zone_name, wall: 'Anterior' | 'Septal' | 'Inferior' | 'Lateral', mass_pct}`

### Relationships (Edges)
- `(:Patient)-[:HAS_VASCULAR_TREE]->(:ArterySegment)`
- `(:Patient)-[:HAS_BIOMARKERS]->(:BiomarkerProfile)`
- `(:ArterySegment)-[:FLOWS_INTO {resistance: Float}]->(:ArterySegment)`
- `(:ArterySegment)-[:CONTAINS_PLAQUE]->(:Plaque)`
- `(:Plaque)-[:ADJACENT_TO {distance_mm: Float}]->(:Bifurcation)`
- `(:BiomarkerProfile)-[:DESTABILIZES {hazard_multiplier: Float}]->(:Plaque)`
- `(:ArterySegment)-[:PERFUSES]->(:MyocardiumZone)`

---

## 5. FalkorDB Graph Algorithms as Agent Tools

1. **Downstream Ischemic Blast Radius (`shortestPath` & Reachability Traversal)**:
   - When a plaque is identified, the agent queries all downstream paths from that segment to determine which `MyocardiumZone` nodes will undergo oxygen starvation if occluded.
2. **Critical Bifurcation Chokepoints (`Betweenness Centrality`)**:
   - Identifies high-centrality vascular junctions that sustain the greatest systemic flow and represent single points of failure (SPOF) in the patient's heart.
3. **Multi-Hop Vulnerability Detection**:
   - Traverses: `(Biomarkers) -> (Plaque) -> (Bifurcation Shear)` to flag low-stenosis soft plaques experiencing high physical shear and biological cap erosion.

---

## 6. System Architecture & Tech Stack

```text
┌─────────────────────────────────────────────────────────────┐
│                 Frontend (Next.js / Vite)                   │
│   - Interactive 2D/3D Coronary Tree (Cytoscape / React Flow)│
│   - Real-Time Path Highlighting & Ischemia Heatmap         │
└──────────────────────────────▲──────────────────────────────┘
                               │ REST / WebSocket
┌──────────────────────────────▼──────────────────────────────┐
│                Agent Core (Python / FastAPI)                │
│   - Model Context Protocol (MCP) Server                     │
│   - LangGraph / PydanticAI Agent Reasoner                   │
│   - LLM: Claude 3.5 Sonnet / GPT-4o / Groq Llama 3          │
└──────────────────────────────▲──────────────────────────────┘
                               │ Cypher Queries / Port 6379
┌──────────────────────────────▼──────────────────────────────┐
│                  FalkorDB (Docker Container)                │
│   - GraphBLAS Linear Algebra Traversal Engine               │
│   - Multi-tenant isolated graphs per patient                │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Hackathon Execution Roadmap (Oct 15–18, 2026)

- **Phase 1 (Setup & Data)**: Launch FalkorDB in Docker. Seed with 25 realistic patient coronary tree graphs (healthy, moderate stenosis, and vulnerable soft plaque scenarios).
- **Phase 2 (MCP Agent Backend)**: Build FastAPI server with MCP tools (`analyze_patient_vascular_tree`, `calculate_ischemic_blast_radius`, `find_vulnerable_plaques`).
- **Phase 3 (Visual Frontend)**: Build interactive vascular tree canvas where doctors can click artery branches, simulate plaque rupture, and watch threatened myocardium zones glow red.
- **Phase 4 (Deliverables)**: Record high-impact 3-minute demo video and publish the technical blog post for the AirPods 5 side prize.
