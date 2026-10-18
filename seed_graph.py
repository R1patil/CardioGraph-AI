"""
CardioGraph AI — FalkorDB Vascular Graph Seeder
================================================
Populates the complete coronary arterial tree, hemodynamic bifurcation metrics,
plaque phenotypes, biomarker profiles, and myocardial perfusion relationships
into FalkorDB using Cypher.

Features:
- Connects to FalkorDB Cloud or Local FalkorDB instance via environment variables or CLI.
- Automatically handles indexes on Patient, ArterySegment, Bifurcation, Plaque, MyocardiumZone.
- Generates canonical vascular network + patient-specific lesions.
- Includes verification Cypher queries demonstrating the Rupture Paradox and Reachability Blast Radius.
- Supports `--dry-run` and `--export-cypher` to generate seed_queries.cypher.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataset import generate_cohort, export_dataset_to_dict

try:
    import falkordb
    from falkordb import FalkorDB, Graph
    FALKORDB_AVAILABLE = True
except ImportError:
    FALKORDB_AVAILABLE = False


# ============================================================================
# FALKORDB CONNECTION MANAGER
# ============================================================================

def get_falkordb_connection(
    host: Optional[str] = None,
    port: Optional[int] = None,
    password: Optional[str] = None,
    username: Optional[str] = None,
    ssl: Optional[bool] = None,
    url: Optional[str] = None
) -> Tuple[Optional[FalkorDB], Optional[str]]:
    """Establishes connection to FalkorDB instance (Cloud or Local)."""
    if not FALKORDB_AVAILABLE:
        return None, "falkordb library is not installed."

    host = host or os.getenv("FALKORDB_HOST", "localhost")
    port = port or int(os.getenv("FALKORDB_PORT", "6379"))
    password = password or os.getenv("FALKORDB_PASSWORD", None)
    username = username or os.getenv("FALKORDB_USERNAME", None)
    url = url or os.getenv("FALKORDB_URL", None)

    if ssl is None:
        ssl_env = os.getenv("FALKORDB_SSL", "false").lower()
        ssl = ssl_env in ("true", "1", "yes")

    try:
        if url:
            client = FalkorDB.from_url(url)
        else:
            client = FalkorDB(
                host=host,
                port=port,
                password=password,
                username=username,
                ssl=ssl
            )
        # Test connection by selecting or pinging
        client.connection.ping()
        return client, None
    except Exception as e:
        return None, f"Could not connect to FalkorDB at {host}:{port} — {str(e)}"


# ============================================================================
# CYPHER STATEMENT GENERATOR
# ============================================================================

def build_schema_indices_cypher() -> List[str]:
    """Creates indexes for rapid traversal and lookups in FalkorDB."""
    return [
        "CREATE INDEX FOR (p:Patient) ON (p.id);",
        "CREATE INDEX FOR (s:ArterySegment) ON (s.id);",
        "CREATE INDEX FOR (b:Bifurcation) ON (b.id);",
        "CREATE INDEX FOR (plq:Plaque) ON (plq.id);",
        "CREATE INDEX FOR (z:MyocardiumZone) ON (z.id);",
        "CREATE INDEX FOR (bio:BiomarkerProfile) ON (bio.id);",
    ]


def build_canonical_topology_cypher(topology: Dict[str, Any]) -> List[str]:
    """Generates Cypher statements for artery segments, bifurcations, and myocardium zones."""
    queries: List[str] = []

    # 1. Myocardium Zones
    for z in topology["myocardium_zones"]:
        aha_str = json.dumps(z["aha_segment_ids"])
        q = (
            f"MERGE (z:MyocardiumZone {{id: '{z['id']}'}}) "
            f"SET z.zone_name = '{z['zone_name']}', "
            f"z.wall = '{z['wall']}', "
            f"z.mass_pct = {z['mass_pct']}, "
            f"z.aha_segment_ids = '{aha_str}';"
        )
        queries.append(q)

    # 2. Artery Segments
    for s in topology["artery_segments"]:
        q = (
            f"MERGE (s:ArterySegment {{id: '{s['id']}'}}) "
            f"SET s.name = '{s['name']}', "
            f"s.vessel_system = '{s['vessel_system']}', "
            f"s.diameter_mm = {s['diameter_mm']}, "
            f"s.length_mm = {s['length_mm']}, "
            f"s.baseline_flow_ml_min = {s['baseline_flow_ml_min']}, "
            f"s.resistance = {s['resistance']};"
        )
        queries.append(q)

    # 3. Bifurcations
    for b in topology["bifurcations"]:
        q = (
            f"MERGE (b:Bifurcation {{id: '{b['id']}'}}) "
            f"SET b.name = '{b['name']}', "
            f"b.bifurcation_angle_deg = {b['bifurcation_angle_deg']}, "
            f"b.shear_stress_index = {b['shear_stress_index']}, "
            f"b.turbulence_factor = {b['turbulence_factor']}, "
            f"b.flow_ratio = {b['flow_ratio']};"
        )
        queries.append(q)

        # Connect parent vessel to bifurcation
        queries.append(
            f"MATCH (s:ArterySegment {{id: '{b['parent_vessel']}'}}), (b:Bifurcation {{id: '{b['id']}'}}) "
            f"MERGE (s)-[:LEADS_TO_BIFURCATION]->(b);"
        )

        # Connect bifurcation to daughter vessels
        for daughter in b["daughter_vessels"]:
            queries.append(
                f"MATCH (b:Bifurcation {{id: '{b['id']}'}}), (d:ArterySegment {{id: '{daughter}'}}) "
                f"MERGE (b)-[:BRANCHES_INTO]->(d);"
            )

    # 4. Directed Hemodynamic Flow Edges (ArterySegment -> ArterySegment)
    for fe in topology["flow_edges"]:
        q = (
            f"MATCH (src:ArterySegment {{id: '{fe['source_segment_id']}'}}), "
            f"(tgt:ArterySegment {{id: '{fe['target_segment_id']}'}}) "
            f"MERGE (src)-[r:FLOWS_INTO]->(tgt) "
            f"SET r.resistance = {fe['resistance']}, r.flow_split_pct = {fe['flow_split_pct']};"
        )
        queries.append(q)

    # 5. Myocardial Perfusion Edges (ArterySegment -> MyocardiumZone)
    for pe in topology["perfusion_edges"]:
        q = (
            f"MATCH (s:ArterySegment {{id: '{pe['artery_segment_id']}'}}), "
            f"(z:MyocardiumZone {{id: '{pe['zone_id']}'}}) "
            f"MERGE (s)-[r:PERFUSES]->(z) "
            f"SET r.perfusion_weight = {pe['perfusion_weight']};"
        )
        queries.append(q)

    return queries


def build_patient_cypher(patient_data: Dict[str, Any]) -> List[str]:
    """Generates Cypher statements for a specific patient, biomarkers, and plaques."""
    queries: List[str] = []
    pid = patient_data["id"]

    clean_presentation = patient_data['clinical_presentation'].replace("'", "\\'")
    benchmark_type = patient_data.get('benchmark_type') or 'STANDARD'
    q_patient = (
        f"MERGE (p:Patient {{id: '{pid}'}}) "
        f"SET p.name = '{patient_data['name']}', "
        f"p.age = {patient_data['age']}, "
        f"p.gender = '{patient_data['gender']}', "
        f"p.smoker = {str(patient_data['smoker']).lower()}, "
        f"p.diabetes = {str(patient_data['diabetes']).lower()}, "
        f"p.hypertension = {str(patient_data['hypertension']).lower()}, "
        f"p.clinical_presentation = '{clean_presentation}', "
        f"p.benchmark_type = '{benchmark_type}';"
    )
    queries.append(q_patient)

    # Connect Patient to the vascular tree root
    queries.append(
        f"MATCH (p:Patient {{id: '{pid}'}}), (lm:ArterySegment {{id: 'SEG_LM'}}), (rca:ArterySegment {{id: 'SEG_RCA_PROX'}}) "
        f"MERGE (p)-[:HAS_VASCULAR_TREE]->(lm) "
        f"MERGE (p)-[:HAS_VASCULAR_TREE]->(rca);"
    )

    # 2. Biomarkers Node
    bio = patient_data.get("biomarkers")
    if bio:
        q_bio = (
            f"MERGE (bio:BiomarkerProfile {{id: '{bio['id']}'}}) "
            f"SET bio.patient_id = '{pid}', "
            f"bio.apob_mg_dl = {bio['apob_mg_dl']}, "
            f"bio.lpa_nmol_l = {bio['lpa_nmol_l']}, "
            f"bio.hscrp_mg_l = {bio['hscrp_mg_l']}, "
            f"bio.ldl_c_mg_dl = {bio['ldl_c_mg_dl']}, "
            f"bio.troponin_i_ng_ml = {bio['troponin_i_ng_ml']}, "
            f"bio.hba1c_pct = {bio['hba1c_pct']};"
        )
        queries.append(q_bio)

        queries.append(
            f"MATCH (p:Patient {{id: '{pid}'}}), (bio:BiomarkerProfile {{id: '{bio['id']}'}}) "
            f"MERGE (p)-[:HAS_BIOMARKERS]->(bio);"
        )

    # 3. Plaques
    for plq in patient_data.get("plaques", []):
        plq_id = plq["id"]
        q_plq = (
            f"MERGE (plq:Plaque {{id: '{plq_id}'}}) "
            f"SET plq.patient_id = '{pid}', "
            f"plq.artery_segment_id = '{plq['artery_segment_id']}', "
            f"plq.stenosis_pct = {plq['stenosis_pct']}, "
            f"plq.plaque_type = '{plq['plaque_type']}', "
            f"plq.cap_thickness_um = {plq['cap_thickness_um']}, "
            f"plq.core_lipid_ratio = {plq['core_lipid_ratio']}, "
            f"plq.calcification_score = {plq['calcification_score']}, "
            f"plq.vulnerability_score = {plq['vulnerability_score']}, "
            f"plq.rupture_risk_category = '{plq['rupture_risk_category']}';"
        )
        queries.append(q_plq)

        # ArterySegment contains Plaque
        queries.append(
            f"MATCH (s:ArterySegment {{id: '{plq['artery_segment_id']}'}}), (plq:Plaque {{id: '{plq_id}'}}) "
            f"MERGE (s)-[:CONTAINS_PLAQUE]->(plq);"
        )

        # Biomarkers destabilize Plaque
        if bio:
            hazard = round((bio["apob_mg_dl"] / 100.0) * (1.0 + bio["hscrp_mg_l"] / 3.0), 2)
            queries.append(
                f"MATCH (bio:BiomarkerProfile {{id: '{bio['id']}'}}), (plq:Plaque {{id: '{plq_id}'}}) "
                f"MERGE (bio)-[r:DESTABILIZES]->(plq) "
                f"SET r.hazard_multiplier = {hazard};"
            )

        # Plaque adjacent to Bifurcation
        if plq.get("adjacent_bifurcation_id") and plq.get("distance_to_bifurcation_mm") is not None:
            queries.append(
                f"MATCH (plq:Plaque {{id: '{plq_id}'}}), (b:Bifurcation {{id: '{plq['adjacent_bifurcation_id']}'}}) "
                f"MERGE (plq)-[r:ADJACENT_TO]->(b) "
                f"SET r.distance_mm = {plq['distance_to_bifurcation_mm']};"
            )

    return queries


def generate_all_cypher_queries(dataset: Dict[str, Any], limit_patients: Optional[int] = None) -> List[str]:
    """Generates the entire sequence of Cypher queries needed to seed FalkorDB."""
    all_queries: List[str] = []

    # 1. Indexes
    all_queries.extend(build_schema_indices_cypher())

    # 2. Canonical Topology
    all_queries.extend(build_canonical_topology_cypher(dataset["canonical_topology"]))

    # 3. Patient Graphs
    patients = dataset["patients"]
    if limit_patients:
        patients = patients[:limit_patients]

    for p in patients:
        all_queries.extend(build_patient_cypher(p))

    return all_queries


# ============================================================================
# SEED EXECUTION & VERIFICATION
# ============================================================================

def seed_to_falkordb(
    graph_name: str = "cardiograph",
    host: Optional[str] = None,
    port: Optional[int] = None,
    password: Optional[str] = None,
    username: Optional[str] = None,
    ssl: Optional[bool] = None,
    limit_patients: Optional[int] = None
) -> bool:
    """Executes the Cypher queries against a live FalkorDB instance."""
    print(f"\n[+] Connecting to FalkorDB...")
    client, err = get_falkordb_connection(host=host, port=port, password=password, username=username, ssl=ssl)
    if err:
        print(f"[!] FalkorDB Connection Failed: {err}")
        return False

    print(f"[+] Connected to FalkorDB! Selecting graph '{graph_name}'...")
    g: Graph = client.select_graph(graph_name)

    # Load dataset
    cohort = generate_cohort(num_patients=25)
    dataset = export_dataset_to_dict(cohort)
    queries = generate_all_cypher_queries(dataset, limit_patients=limit_patients)

    print(f"[+] Executing {len(queries)} Cypher statements into graph '{graph_name}'...")
    success_count = 0
    for i, q in enumerate(queries, 1):
        try:
            g.query(q)
            success_count += 1
            if i % 25 == 0 or i == len(queries):
                print(f"    - Processed {i}/{len(queries)} queries...")
        except Exception as q_err:
            print(f"    [!] Query error on #{i}: {q_err}\n        Query: {q}")

    print(f"\n[+] Seeding complete: {success_count}/{len(queries)} queries executed successfully.")

    # Run verification queries
    run_verification_queries(g)
    return True


def run_verification_queries(graph: Graph):
    """Demonstrates graph algorithms, Rupture Paradox, and blast radius directly in Cypher."""
    print("\n" + "=" * 80)
    print("VERIFYING FALKORDB GRAPH WITH CLINICAL QUERIES")
    print("=" * 80)

    # 1. Total Node Counts
    try:
        res = graph.query("MATCH (n) RETURN labels(n) as label, count(n) as count")
        print("\n[Node Counts by Label]:")
        for row in res.result_set:
            print(f"  {row[0]}: {row[1]}")
    except Exception as e:
        print(f"  [!] Count query error: {e}")

    # 2. Rupture Paradox Query: Compare PAT-101 (stable) vs PAT-102 (vulnerable)
    try:
        query_rupture = """
        MATCH (p:Patient)-[:HAS_BIOMARKERS]->(bio:BiomarkerProfile),
              (p)-[:HAS_VASCULAR_TREE]->(root),
              (s:ArterySegment)-[:CONTAINS_PLAQUE]->(plq:Plaque)
        WHERE plq.patient_id = p.id AND p.id IN ['PAT-101', 'PAT-102']
        OPTIONAL MATCH (plq)-[adj:ADJACENT_TO]->(b:Bifurcation)
        RETURN p.id AS patient_id,
               p.name AS name,
               s.id AS segment,
               plq.stenosis_pct AS stenosis,
               plq.plaque_type AS type,
               plq.cap_thickness_um AS cap_um,
               plq.vulnerability_score AS vulnerability,
               plq.rupture_risk_category AS risk,
               bio.apob_mg_dl AS apob,
               bio.hscrp_mg_l AS hscrp,
               b.name AS adjacent_bifurcation,
               adj.distance_mm AS distance_to_bif_mm
        """
        res = graph.query(query_rupture)
        print("\n[Clinical Demonstration: The Rupture Paradox in FalkorDB]:")
        for row in res.result_set:
            print(f"  Patient: {row[0]} ({row[1]}) | Segment: {row[2]}")
            print(f"    - Stenosis: {row[3]}% | Type: {row[4]} | Cap: {row[5]} µm")
            print(f"    - Vulnerability Score: {row[6]} [{row[7]}]")
            print(f"    - DeepMind ApoB: {row[8]} mg/dL | hs-CRP: {row[9]} mg/L")
            print(f"    - Adjacent Bifurcation: {row[10]} ({row[11]} mm)")
    except Exception as e:
        print(f"  [!] Rupture paradox query error: {e}")

    # 3. Downstream Reachability & Ischemic Blast Radius Query for PAT-102
    try:
        query_blast = """
        MATCH (plq:Plaque {id: 'PLQ-102-LAD-PROX'})<-[:CONTAINS_PLAQUE]-(origin:ArterySegment)
        MATCH (origin)-[:FLOWS_INTO*0..5]->(downstream:ArterySegment)-[:PERFUSES]->(z:MyocardiumZone)
        RETURN DISTINCT z.zone_name AS threatened_zone, z.wall AS wall, z.mass_pct AS mass_pct
        ORDER BY z.mass_pct DESC
        """
        res = graph.query(query_blast)
        print("\n[Downstream Ischemic Blast Radius for PAT-102 (LAD Proximal Rupture)]:")
        total_mass = 0.0
        for row in res.result_set:
            total_mass += float(row[2])
            print(f"  - Zone: {row[0]} (Wall: {row[1]}) -> {row[2]}% LV mass")
        print(f"  ==> TOTAL MYOCARDIUM AT RISK: {round(total_mass, 1)}% OF HEART MUSCLE")
    except Exception as e:
        print(f"  [!] Blast radius query error: {e}")


# ============================================================================
# CLI DISPATCHER
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CardioGraph AI — FalkorDB Vascular Graph Seeder")
    parser.add_argument("--host", default=os.getenv("FALKORDB_HOST", "localhost"), help="FalkorDB Host")
    parser.add_argument("--port", type=int, default=int(os.getenv("FALKORDB_PORT", "6379")), help="FalkorDB Port")
    parser.add_argument("--password", default=os.getenv("FALKORDB_PASSWORD", None), help="FalkorDB Password")
    parser.add_argument("--username", default=os.getenv("FALKORDB_USERNAME", None), help="FalkorDB Username")
    parser.add_argument("--ssl", action="store_true", help="Enable SSL connection for FalkorDB Cloud")
    parser.add_argument("--graph", default="cardiograph", help="FalkorDB Graph Name (default: cardiograph)")
    parser.add_argument("--patients", type=int, default=25, help="Number of patients to seed (default: 25)")
    parser.add_argument("--export-cypher", type=str, default="seed_queries.cypher", help="Export Cypher queries to file")
    parser.add_argument("--dry-run", action="store_true", help="Generate Cypher statements without connecting to database")

    args = parser.parse_args()

    print("=" * 80)
    print("CardioGraph AI — FalkorDB Seeder")
    print(f"Target Graph: '{args.graph}' | Patients to Seed: {args.patients}")
    print("=" * 80)

    # Load or generate cohort dataset
    cohort = generate_cohort(num_patients=args.patients)
    dataset = export_dataset_to_dict(cohort)
    queries = generate_all_cypher_queries(dataset, limit_patients=args.patients)

    # Always write export file so users have complete offline visibility
    out_file = Path(args.export_cypher)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("// CardioGraph AI — FalkorDB Seed Cypher Statements\n")
        f.write(f"// Generated {len(queries)} statements for graph '{args.graph}'\n\n")
        for q in queries:
            f.write(f"{q}\n")
    print(f"[+] Exported {len(queries)} Cypher statements to: {out_file.resolve()} ({round(out_file.stat().st_size / 1024, 1)} KB)")

    if args.dry_run:
        print("\n[i] Dry-run mode completed. Cypher queries are ready for execution.")
        return

    # Attempt live connection
    connected = seed_to_falkordb(
        graph_name=args.graph,
        host=args.host,
        port=args.port,
        password=args.password,
        username=args.username,
        ssl=args.ssl,
        limit_patients=args.patients
    )

    if not connected:
        print("\n" + "-" * 80)
        print("[i] Quick Guide to Run FalkorDB:")
        print("  Option A (FalkorDB Cloud Free Tier):")
        print("    1. Sign up at https://falkordb.com/cloud (free)")
        print("    2. Set environment variables:")
        print("       set FALKORDB_HOST=<cloud-endpoint>")
        print("       set FALKORDB_PORT=<cloud-port>")
        print("       set FALKORDB_PASSWORD=<cloud-password>")
        print("       set FALKORDB_SSL=true")
        print("    3. Run: python seed_graph.py")
        print("\n  Option B (Docker Desktop or Local Container):")
        print("    docker run -p 6379:6379 -it --rm falkordb/falkordb:latest")
        print("    python seed_graph.py")
        print("\n  Option C (Offline Cypher Query Inspection):")
        print(f"    View generated Cypher file: {out_file.name}")
        print("-" * 80)


if __name__ == "__main__":
    main()
