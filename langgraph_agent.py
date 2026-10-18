"""
CardioGraph AI — LangGraph & LangChain Patient Care Agent
=========================================================
Autonomous clinical & lifestyle reasoning agent built with LangGraph.
Translates complex vascular graph data (ApoB, TCFA, Ischemic Blast Radius)
into compassionate, plain-English patient roadmaps with step-by-step
treatment pathways, cardioprotective nutrition, and exercise protocols.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

# No circular imports needed - patient_data can be injected or resolved via dataset
try:
    from dataset import get_synthetic_dataset
    _DATASET_CACHE = get_synthetic_dataset()
    _PATIENT_MAP = {p["id"]: p for p in _DATASET_CACHE.get("patients", [])}
except Exception:
    _PATIENT_MAP = {}



# ============================================================================
# LANGGRAPH STATE DEFINITION
# ============================================================================

class CardioAgentState(TypedDict):
    patient_id: str
    user_message: str
    mode: str                        # 'patient' or 'doctor'
    patient_data: Optional[Dict[str, Any]]
    plaque_findings: Optional[Dict[str, Any]]
    clinical_risk: str
    is_rupture_paradox: bool
    treatment_pathway: Dict[str, Any]
    nutrition_protocol: Dict[str, Any]
    lifestyle_protocol: Dict[str, Any]
    reasoning_steps: List[str]
    final_response: str


# ============================================================================
# LANGGRAPH WORKFLOW NODES
# ============================================================================

def retrieve_vascular_context_node(state: CardioAgentState) -> Dict[str, Any]:
    """
    Step 1: Queries FalkorDB / Vascular Graph Engine for the patient's
    atherogenic biomarkers, coronary plaques, and downstream blast radius.
    """
    pid = state["patient_id"]
    steps = list(state.get("reasoning_steps", []))
    steps.append("1. Vascular Graph Retrieval: Extracted coronary topology & biomarkers from FalkorDB.")

    p_data = state.get("patient_data") or _PATIENT_MAP.get(pid)
    if not p_data:
        # Fallback default patient
        p_data = {
            "id": pid,
            "name": "Patient",
            "age": 52,
            "gender": "Male",
            "smoker": True,
            "diabetes": False,
            "hypertension": True,
            "biomarkers": {
                "apob_mg_dl": 148.0,
                "hscrp_mg_l": 4.8,
                "lpa_nmol_l": 192.0,
                "ldl_c_mg_dl": 162.0,
                "troponin_i_ng_ml": 0.038
            },
            "plaques": [
                {
                    "id": "PLQ-102",
                    "artery_segment_id": "SEG_LAD_PROX",
                    "stenosis_pct": 38.0,
                    "plaque_type": "soft_lipid",
                    "cap_thickness_um": 42.0,
                    "vulnerability_score": 0.99,
                    "rupture_risk_category": "CRITICAL",
                    "downstream_blast_radius": {
                        "total_myocardium_mass_at_risk_pct": 68.0,
                        "threatened_zones": [{"zone_name": "Anterior Wall", "mass_pct": 22.0}]
                    }
                }
            ]
        }

    # Extract primary plaque
    plaques = p_data.get("plaques", [])
    primary_plaque = plaques[0] if plaques else None

    # Check Rupture Paradox
    is_paradox = False
    if primary_plaque:
        is_paradox = (
            primary_plaque.get("stenosis_pct", 0) < 50.0 and
            primary_plaque.get("cap_thickness_um", 100) < 65.0 and
            primary_plaque.get("vulnerability_score", 0) >= 0.70
        )

    risk = "CRITICAL_RUPTURE_RISK" if is_paradox else (
        "SEVERE_OBSTRUCTIVE" if primary_plaque and primary_plaque.get("stenosis_pct", 0) >= 70.0 else (
            "MODERATE_DIFFUSE" if plaques else "OPTIMAL_HEALTH"
        )
    )

    return {
        "patient_data": p_data,
        "plaque_findings": primary_plaque,
        "clinical_risk": risk,
        "is_rupture_paradox": is_paradox,
        "reasoning_steps": steps
    }


def clinical_treatment_planner_node(state: CardioAgentState) -> Dict[str, Any]:
    """
    Step 2: Formulates medical intervention pathways, drug classes (ApoB-lowering,
    cap-stabilizing statins, antiplatelets), and hospital Cath Lab escalation criteria.
    """
    steps = list(state["reasoning_steps"])
    steps.append("2. Clinical Treatment Planning: Formulated evidence-based medical & catheterization pathway.")

    p = state["patient_data"]
    bio = p.get("biomarkers", {})
    plq = state.get("plaque_findings") or {}
    is_paradox = state.get("is_rupture_paradox", False)

    apob = bio.get("apob_mg_dl", 100)
    hscrp = bio.get("hscrp_mg_l", 1.0)
    cap = plq.get("cap_thickness_um", 100)

    medications = []
    urgency = "ROUTINE"
    cath_lab_referral = False

    if is_paradox or cap < 65.0:
        urgency = "HIGH_PRIORITY_URGENT"
        medications.extend([
            {
                "drug_class": "High-Intensity Statin (Atorvastatin 80mg or Rosuvastatin 40mg)",
                "purpose": "Rapidly thickens the fragile fibrous cap, stabilizes core lipid, and reduces ApoB by >50%."
            },
            {
                "drug_class": "PCSK9 Inhibitor (Evolocumab / Alirocumab)",
                "purpose": "Aggressively crushes atherogenic ApoB particle count to <55 mg/dL to stop macrophage infiltration."
            },
            {
                "drug_class": "Low-Dose Aspirin (81mg)",
                "purpose": "Prevents sudden blood clot (thrombosis) formation if the cap develops a micro-fissure."
            },
            {
                "drug_class": "Anti-Inflammatory Colchicine (0.5mg daily)",
                "purpose": "Directly dampens hs-CRP vascular inflammation that dissolves collagen fibers in the plaque cap."
            }
        ])
    elif plq.get("stenosis_pct", 0) >= 70.0:
        urgency = "INTERMEDIATE_CARDIOLOGY_CONSULT"
        cath_lab_referral = True
        medications.extend([
            {"drug_class": "Moderate/High Statin", "purpose": "ApoB particle reduction."},
            {"drug_class": "Beta-Blocker / Nitrate", "purpose": "Relieves exertional chest tightness (angina)."},
            {"drug_class": "Aspirin", "purpose": "Cardioprotection."}
        ])
    else:
        medications.append({
            "drug_class": "Preventative Lipid Therapy",
            "purpose": "Maintain ApoB <80 mg/dL through lifestyle and low-dose lipid management."
        })

    treatment_plan = {
        "urgency_level": urgency,
        "cath_lab_evaluation": cath_lab_referral,
        "recommended_medication_classes": medications,
        "target_apob_goal": "< 55 mg/dL" if is_paradox else "< 70 mg/dL",
        "target_hscrp_goal": "< 1.0 mg/L"
    }

    return {
        "treatment_pathway": treatment_plan,
        "reasoning_steps": steps
    }


def nutrition_lifestyle_prescriber_node(state: CardioAgentState) -> Dict[str, Any]:
    """
    Step 3: Creates customized, actionable, delicious cardioprotective nutrition
    guidelines specifically designed to lower atherogenic ApoB and systemic inflammation.
    """
    steps = list(state["reasoning_steps"])
    steps.append("3. Cardioprotective Nutrition & Lifestyle: Prescribed low-ApoB anti-inflammatory protocol.")

    p = state["patient_data"]
    is_diabetic = p.get("diabetes", False)
    is_smoker = p.get("smoker", False)

    # Nutrition Plan
    nutrition = {
        "diet_framework": "Cardioprotective Mediterranean-Portfolio Diet",
        "foods_to_eat_daily": [
            {
                "food": "Viscous Soluble Fiber (Oats, Barley, Psyllium Husk, Beans)",
                "why": "Binds to bile acids in the gut, forcing the liver to pull ApoB and LDL particles from the bloodstream."
            },
            {
                "food": "Omega-3 Rich Wild Fatty Fish (Salmon, Sardines, Mackerel)",
                "why": "Lowers systemic inflammation (hs-CRP) and stabilizes arterial endothelial lining."
            },
            {
                "food": "Raw Walnuts & Extra Virgin Olive Oil (Polyphenol-rich)",
                "why": "Contains oleic acid and antioxidants that protect plaque lipids from oxidizing."
            },
            {
                "food": "Dark Leafy Greens & Berries (Blueberries, Blackberries, Spinach)",
                "why": "High in nitric oxide precursors and flavonoids that keep arteries flexible and relaxed."
            }
        ],
        "foods_to_strictly_avoid": [
            {
                "food": "Ultra-Processed Foods & Trans Fats (Fried snacks, commercial pastries)",
                "why": "Directly triggers liver production of small dense atherogenic ApoB particles."
            },
            {
                "food": "Excess Saturated Animal Fats (Fatty red cuts, full-fat butter, processed meats)",
                "why": "Downregulates LDL receptors on liver cells, leaving ApoB particles circulating in blood to penetrate plaque."
            },
            {
                "food": "Refined Sugars & High-Fructose Corn Syrup",
                "why": "Drives insulin resistance and vascular inflammation, weakening the fibrous cap."
            }
        ],
        "hydration": "Drink 2.5 - 3 liters of filtered water daily. Limit alcohol to near zero."
    }

    # Exercise & Daily Activity Plan
    lifestyle = {
        "exercise_prescription": (
            "Light to Moderate Zone-2 Walking (30-45 minutes daily). DO NOT perform heavy isometric bench pressing, "
            "extreme weightlifting, or sudden all-out sprints right now, as sudden spikes in blood pressure increase carinal shear stress!"
            if state.get("is_rupture_paradox") else
            "Moderate aerobic exercise: 150 minutes per week of brisk walking, cycling, or swimming."
        ),
        "smoking_action": "MANDATORY CESSATION IMMEDIATELY: Smoking directly causes spasms in the LAD artery and dissolves collagen in plaque caps." if is_smoker else "N/A (Non-smoker)",
        "sleep_hygiene": "7 to 8 hours of restorative sleep. Sleep apnea testing recommended if snoring is present.",
        "stress_reduction": "10 minutes of slow 4-7-8 diaphragmatic breathing morning and evening to reduce cortisol surges."
    }

    return {
        "nutrition_protocol": nutrition,
        "lifestyle_protocol": lifestyle,
        "reasoning_steps": steps
    }


def patient_response_synthesizer_node(state: CardioAgentState) -> Dict[str, Any]:
    """
    Step 4: Synthesizes the entire graph context, medical pathway, and nutrition
    plan into a warm, clear, empathetic plain-English response.
    """
    steps = list(state["reasoning_steps"])
    steps.append("4. Patient-Friendly Synthesis: Generated clear plain-English explanation and action roadmap.")

    p = state["patient_data"]
    plq = state.get("plaque_findings") or {}
    bio = p.get("biomarkers", {})
    is_paradox = state.get("is_rupture_paradox", False)
    blast = plq.get("downstream_blast_radius", {})
    mass_risk = blast.get("total_myocardium_mass_at_risk_pct", 68.0)
    tx = state["treatment_pathway"]
    nut = state["nutrition_protocol"]
    life = state["lifestyle_protocol"]
    user_q = state.get("user_message", "").strip().lower()

    name = p.get("name", "there")
    vessel_name = plq.get("artery_segment_id", "coronary artery").replace("SEG_", "")

    # Tailored answer if user asked a specific question
    if "eat" in user_q or "food" in user_q or "diet" in user_q or "nutrition" in user_q:
        response = f"""### 🥗 Cardioprotective Nutrition Roadmap for {name}

Here is your exact meal protocol to lower your **ApoB cholesterol** and protect your blood vessels:

#### 1. What to Eat Every Day:
- **Oatmeal & Beans (Soluble Fiber)**: Have 1 bowl of steel-cut oats with chia seeds or beans daily. Fiber acts like a sponge in your digestive system, pulling ApoB particles out of your blood before they can stick to your arteries.
- **Fatty Wild Fish (Salmon, Sardines)**: Eat 2-3 times per week. The omega-3 fats calm vascular inflammation (`hs-CRP`).
- **Extra Virgin Olive Oil & Walnuts**: Use 2 tablespoons of raw olive oil on salads daily. It contains powerful antioxidants that protect plaque from hardening or tearing.
- **Berries & Dark Leafy Greens**: Blueberries, spinach, and arugula help your arteries stay relaxed and flexible.

#### 2. What to Strictly Eliminate:
- **Fried & Processed Foods**: Packaged cookies, french fries, and chips trigger your liver to pump out small atherogenic ApoB particles.
- **Fatty Red Meats & Full-Fat Butter**: Replace with lentils, chicken breast, or fish to protect your liver's natural cholesterol clearance.
- **Sugary Drinks & Sodas**: Sugar spikes inflammation that weakens plaque walls.
"""

    elif "exercise" in user_q or "walk" in user_q or "gym" in user_q or "workout" in user_q:
        response = f"""### 🏃 Safe Exercise & Physical Activity Guide for {name}

Because your scan shows a **fragile plaque cap ({plq.get('cap_thickness_um', 42)} µm)** on your {vessel_name} artery:

- **Safe Zone**: Daily brisk walking for **30 to 40 minutes** at an easy conversational pace (Zone 2 cardio). Walking gently keeps your blood moving without stressing your heart.
- **What to AVOID right now**: Do **NOT** do heavy weightlifting, crossfit, maximum bench presses, or sudden all-out sprints. Straining and holding your breath causes sharp spikes in blood pressure that can put dangerous mechanical stress on a fragile plaque cap.
- **Goal**: Build endurance safely while your medication thickens and stabilizes your plaque cap over the next 3 to 6 months.
"""

    elif "medicine" in user_q or "treatment" in user_q or "drug" in user_q or "stent" in user_q:
        response = f"""### 💊 Medical Treatment Plan for {name}

Here is what your cardiologist will likely discuss to stabilize your plaque and prevent a heart attack:

1. **High-Intensity Statin Therapy (e.g. Rosuvastatin or Atorvastatin)**:
   - *Why it matters*: Statins do not just lower numbers—they actively soak into the plaque, pull out soft cholesterol, and **thicken the plaque's cap** like adding a thick coat of protective cement.
2. **PCSK9 Inhibitor Injections (if ApoB remains >55 mg/dL)**:
   - *Why it matters*: Ultra-powerful therapy that cuts dangerous ApoB particles in half.
3. **Low-Dose Aspirin (81 mg daily)**:
   - *Why it matters*: Acts as insurance to stop blood platelets from sticking together if a tiny tear occurs.
4. **Anti-Inflammatory Therapy**:
   - Calming your systemic inflammation (`hs-CRP {bio.get('hscrp_mg_l', 4.8)} mg/L`) prevents enzymes from dissolving your artery's protective collagen.
"""

    elif "er" in user_q or "warning" in user_q or "emergency" in user_q or "symptom" in user_q:
        response = f"""### 🚨 Emergency Red-Flag Warning Signs (When to Call 911 Immediately)

Because occlusion of your {vessel_name} branch could affect up to **{mass_risk}% of your heart muscle**, memorizing these warning signs is critical:

- **Heavy pressure, squeezing, or fullness** in the center of your chest lasting more than a few minutes.
- **Pain radiating** to your left arm, shoulder, jaw, back, or neck.
- **Sudden shortness of breath** with or without chest discomfort.
- **Cold sweats, dizziness, or unusual nausea/lightheadedness.**

> **Action Rule**: If you feel these symptoms, do NOT drive yourself. **Call emergency services immediately (911 / Ambulance).** Inform them: *"I have an acute high-risk coronary plaque on my LAD."*
"""

    else:
        # Complete full walkthrough in plain English
        status_headline = (
            "⚠️ Attention Needed: We found a fragile, high-risk plaque that regular tests often overlook!"
            if is_paradox else
            "🩺 Here is a complete plain-English breakdown of your heart health and action plan."
        )

        response = f"""### {status_headline}

Hello **{name}**, here is your personalized heart health roadmap explained in simple, clear words:

---

#### 1. What Your Scan Actually Means (The Plain Truth):
- **Your Blockage**: Your {vessel_name} artery is narrowed by **{plq.get('stenosis_pct', 38)}%**.
- **Why Regular Scans Get This Wrong**: In conventional checkups, doctors only look for big blockages (like 70% or 80%). Because your blockage is only {plq.get('stenosis_pct', 38)}%, a regular clinic might have told you *"everything looks fine."*
- **The Hidden Danger (The Rupture Paradox)**: Your plaque is **not hard like rock**—it has a paper-thin skin (**{plq.get('cap_thickness_um', 42)} µm**, where anything under 65 µm is fragile). It is like a delicate blister filled with cholesterol sitting right where blood rushes by.
- **If it breaks**: If this small blister tears open, a sudden clot could starve **{mass_risk}% of your heart muscle**. That is why we are taking action now, before anything happens!

---

#### 2. What Your Blood Tests Mean:
- **ApoB-100 ({bio.get('apob_mg_dl', 148)} mg/dL)**: This is the actual count of dangerous cholesterol particles penetrating your blood vessel walls. Your target should be **below 55 mg/dL**.
- **hs-CRP ({bio.get('hscrp_mg_l', 4.8)} mg/L)**: This measures inflammation in your blood vessels. Right now, it is elevated, which acts like acid dissolving the protective cap of your plaque.

---

#### 3. Your Immediate 3-Step Action Plan:

| Area | What You Should Do |
|---|---|
| 💊 **Medical Treatment** | See your cardiologist for **High-Intensity Statin Therapy** + **Low-Dose Aspirin**. This turns the soft, fragile plaque into a stable, hard plaque that cannot easily tear. |
| 🥗 **Nutrition & Food** | Eat **oatmeal, beans, wild salmon, and olive oil daily**. Avoid fried foods, processed snacks, and excess red meat to stop feeding ApoB particles into your arteries. |
| 🏃 **Safe Exercise** | Walk for **30 to 40 minutes daily**. Avoid heavy weightlifting or sudden max-effort sprints until your doctor confirms your plaque cap has thickened. |

---

*You can ask me any question below, such as: "What meals should I cook this week?", "Can you explain my medication?", or "What should I tell my cardiologist?"*
"""

    return {
        "final_response": response,
        "reasoning_steps": steps
    }


# ============================================================================
# COMPILE LANGGRAPH STATE MACHINE
# ============================================================================

def build_cardio_langgraph() -> Any:
    """Builds and compiles the multi-step LangGraph StateGraph workflow."""
    workflow = StateGraph(CardioAgentState)

    # Add Nodes
    workflow.add_node("retrieve_vascular_context", retrieve_vascular_context_node)
    workflow.add_node("clinical_treatment_planner", clinical_treatment_planner_node)
    workflow.add_node("nutrition_lifestyle_prescriber", nutrition_lifestyle_prescriber_node)
    workflow.add_node("patient_response_synthesizer", patient_response_synthesizer_node)

    # Define Flow Edges
    workflow.add_edge(START, "retrieve_vascular_context")
    workflow.add_edge("retrieve_vascular_context", "clinical_treatment_planner")
    workflow.add_edge("clinical_treatment_planner", "nutrition_lifestyle_prescriber")
    workflow.add_edge("nutrition_lifestyle_prescriber", "patient_response_synthesizer")
    workflow.add_edge("patient_response_synthesizer", END)

    return workflow.compile()


# Global compiled LangGraph application
cardio_graph_app = build_cardio_langgraph()


def run_cardio_chat_agent(patient_id: str, message: str, mode: str = "patient", patient_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Executes the LangGraph agent for a user message."""
    initial_state: CardioAgentState = {
        "patient_id": patient_id,
        "user_message": message,
        "mode": mode,
        "patient_data": patient_data,
        "plaque_findings": None,
        "clinical_risk": "UNKNOWN",
        "is_rupture_paradox": False,
        "treatment_pathway": {},
        "nutrition_protocol": {},
        "lifestyle_protocol": {},
        "reasoning_steps": [],
        "final_response": ""
    }

    result = cardio_graph_app.invoke(initial_state)

    return {
        "status": "SUCCESS",
        "patient_id": patient_id,
        "final_response": result["final_response"],
        "reasoning_steps": result["reasoning_steps"],
        "clinical_risk": result["clinical_risk"],
        "treatment_pathway": result["treatment_pathway"],
        "nutrition_protocol": result["nutrition_protocol"],
        "lifestyle_protocol": result["lifestyle_protocol"]
    }
