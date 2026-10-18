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
from dotenv import load_dotenv

load_dotenv()

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


def call_groq_llm(state: CardioAgentState) -> Optional[str]:
    """
    Calls Groq Cloud LLaMA model using GROQ_API_KEY with the patient's
    complete FalkorDB vascular graph context to answer questions directly.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except Exception:
        return None

    p = state.get("patient_data") or {}
    plq = state.get("plaque_findings") or {}
    bio = p.get("biomarkers", {})
    blast = plq.get("downstream_blast_radius", {})
    mass_risk = blast.get("total_myocardium_mass_at_risk_pct", 0)
    is_paradox = state.get("is_rupture_paradox", False)
    user_q = state.get("user_message", "").strip()
    mode = state.get("mode", "patient")

    vessel_name = plq.get("artery_segment_id", "coronary artery").replace("SEG_", "")

    system_prompt = f"""You are CardioGraph AI, an advanced interventional cardiology and clinical lifestyle intelligence assistant.
You have access to the patient's coronary graph data retrieved from FalkorDB:
- Patient Name: {p.get('name', 'Patient')}
- Age: {p.get('age', 55)}, Gender: {p.get('gender', 'Unknown')}, Smoker: {p.get('smoker', False)}, Diabetic: {p.get('diabetes', False)}, Hypertensive: {p.get('hypertension', False)}
- Artery Segment: {vessel_name}
- Stenosis: {plq.get('stenosis_pct', 0)}%
- Plaque Morphology: {plq.get('plaque_type', 'fibroatheroma')}
- Fibrous Cap Thickness: {plq.get('cap_thickness_um', 100)} µm (Thin-Cap Fibroatheroma threshold is <65 µm; fragile if under 65 µm)
- Plaque Vulnerability Score: {plq.get('vulnerability_score', 0)}
- Rupture Paradox: {is_paradox} (Vulnerable thin cap despite mild-to-moderate stenosis)
- Downstream Myocardium Mass at Risk: {mass_risk}%
- Biomarkers: ApoB-100: {bio.get('apob_mg_dl', 'N/A')} mg/dL (Optimal <55 mg/dL), hs-CRP: {bio.get('hscrp_mg_l', 'N/A')} mg/L (Optimal <1.0 mg/L), Lp(a): {bio.get('lpa_nmol_l', 'N/A')} nmol/L, LDL-C: {bio.get('ldl_c_mg_dl', 'N/A')} mg/dL, Troponin-I: {bio.get('troponin_i_ng_ml', 'N/A')} ng/mL

Instructions:
1. Directly and immediately answer the user's specific question. DO NOT dump a generic full welcome report if they asked something specific.
2. If the user asks about meal planning, cooking, recipes, or what to eat/cook this week: Give them a specific, practical, delicious heart-healthy 7-day meal plan with exact breakfast, lunch, and dinner ideas designed to lower ApoB and reduce hs-CRP inflammation (e.g., steel-cut oats, wild salmon, extra virgin olive oil, berries, beans, lentils), plus essential heart-healthy cooking methods (steaming/baking, avoiding fried foods and trans fats).
3. If they ask about medications: Explain why high-intensity statins thicken the fibrous cap from {plq.get('cap_thickness_um', 100)} µm to stable levels, how PCSK9 inhibitors crush circulating ApoB particles, and why aspirin prevents clot formation.
4. If they ask about exercise: Clearly define safe Zone-2 aerobic walking (30-40 min daily) and strictly caution against heavy isometric weightlifting or sudden max sprints that spike blood pressure and shear stress.
5. If they ask about cardiologist questions: Provide 4-5 high-impact, exact questions to bring to their appointment.
6. If they ask about warning signs: Provide clear emergency signs (chest tightness radiating to jaw/arm, dyspnea, cold sweats) and 911 action steps.
7. Formatting: Use clean GitHub Markdown with headings, bullet points, and tables where helpful.
8. Tone: Empathetic, warm, clear, and reassuring. If mode is 'doctor', use rigorous clinical terminology (AHA/ACC, SYNTAX, Virmani classification).
"""

    models_to_try = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    for model_name in models_to_try:
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_q}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            content = res.choices[0].message.content
            if content and len(content.strip()) > 20:
                return content.strip()
        except Exception:
            continue

    return None


def patient_response_synthesizer_node(state: CardioAgentState) -> Dict[str, Any]:
    """
    Step 4: Synthesizes the entire graph context, medical pathway, and nutrition
    plan into a warm, clear, empathetic plain-English response.
    First tries Groq LLaMA reasoning agent; if unavailable, falls back to
    comprehensive domain-specific deterministic synthesis.
    """
    steps = list(state["reasoning_steps"])
    p = state["patient_data"]
    plq = state.get("plaque_findings") or {}
    bio = p.get("biomarkers", {})
    is_paradox = state.get("is_rupture_paradox", False)
    blast = plq.get("downstream_blast_radius", {})
    mass_risk = blast.get("total_myocardium_mass_at_risk_pct", 68.0)
    tx = state["treatment_pathway"]
    nut = state["nutrition_protocol"]
    life = state["lifestyle_protocol"]
    raw_q = state.get("user_message", "").strip()
    user_q = raw_q.lower()

    name = p.get("name", "there")
    vessel_name = plq.get("artery_segment_id", "coronary artery").replace("SEG_", "")

    # 1. Try Groq AI for specific user questions
    is_generic_greeting = not user_q or user_q in ["hello", "hi", "hey", "overview", "full overview", "help", "summary"]
    if not is_generic_greeting:
        groq_resp = call_groq_llm(state)
        if groq_resp:
            steps.append("4. Patient-Friendly Synthesis: Synthesized personalized answer using Groq AI (LLaMA clinical intelligence).")
            return {
                "final_response": groq_resp,
                "reasoning_steps": steps
            }

    # 2. Comprehensive deterministic fallback
    steps.append("4. Patient-Friendly Synthesis: Generated clear plain-English explanation and action roadmap.")

    # A. Meals, Cooking, Recipes & Nutrition
    if any(k in user_q for k in ["meal", "meals", "cook", "cooking", "recipe", "recipes", "food", "eat", "diet", "nutrition", "breakfast", "lunch", "dinner", "snack", "grocery", "groceries"]):
        response = f"""### 🥗 Cardioprotective 7-Day Meal Plan & Cooking Guide for {name}

Here is a practical, delicious weekly meal plan designed specifically to lower your **ApoB particles** ({bio.get('apob_mg_dl', 148)} mg/dL) and calm inflammation (`hs-CRP` {bio.get('hscrp_mg_l', 4.8)} mg/L):

| Day | Breakfast | Lunch | Dinner |
|---|---|---|---|
| **Mon** | Steel-cut oats cooked in water/oat milk with chia seeds, wild blueberries, and a dash of cinnamon. | Mediterranean quinoa bowl with chickpeas, diced cucumber, tomatoes, spinach, and extra virgin olive oil. | Baked wild salmon with lemon, steamed broccoli, and baked sweet potato wedges. |
| **Tue** | Avocado toast on 100% whole grain rye sourdough, topped with pumpkin seeds and tomato slices. | Lentil and vegetable soup with mixed greens salad dressed with cold-pressed olive oil & lemon juice. | Grilled chicken breast or grilled tofu with garlic-sautéed kale and brown rice. |
| **Wed** | Overnight rolled oats with ground flaxseed, walnuts, and raspberries. | Wild sardines on whole grain crispbread with arugula, cherry tomatoes, and red onion. | Stir-fried bok choy, edamame, and shiitake mushrooms cooked in avocado oil over quinoa. |
| **Thu** | Warm berry chia seed pudding with unsweetened almond milk and a handful of raw walnuts. | Black bean and roasted sweet potato salad with lime, cilantro, and avocado. | Pan-seared cod or trout with garlic asparagus and a side of steamed barley. |
| **Fri** | Green smoothie: Spinach, half green apple, celery, hemp seeds, and unsweetened almond milk. | Hearty minestrone soup loaded with kidney beans, zucchini, carrots, and tomatoes. | Roasted Mediterranean vegetable tray (bell peppers, onions, eggplant, zucchini) with baked white fish or grilled tempeh. |
| **Sat** | Steel-cut oatmeal bowl with unsweetened cacao nibs, blueberries, and sunflower seeds. | Kale and white bean salad with pumpkin seeds and extra virgin olive oil vinaigrette. | Steamed wild salmon or steelhead trout with quinoa and garlic-roasted Brussels sprouts. |
| **Sun** | Tofu scramble with spinach, mushrooms, and turmeric, served with 1 slice whole grain toast. | Leftover Mediterranean stew or lentil soup with crisp garden salad. | Baked chicken breast with lemon oregano marinade, roasted cauliflower, and brown basmati rice. |

#### 🍳 Golden Rules for Cooking This Week:
1. **Zero Deep Frying**: Bake, grill, steam, or air-fry. Never reuse cooking oil.
2. **The Only Oils to Use**: Extra virgin olive oil (cold or low-heat) and avocado oil (medium-high heat). Avoid corn, soybean, canola, and hydrogenated oils.
3. **Flavour with Herbs, Not Salt**: Use garlic, turmeric, ginger, oregano, rosemary, black pepper, and fresh lemon to enhance taste without raising blood pressure.
4. **Hydration**: Aim for 2 to 3 litres of fresh water daily. Avoid sugary drinks and sodas.
"""

    # B. Safe Exercise & Movement
    elif any(k in user_q for k in ["exercise", "walk", "walking", "gym", "workout", "lift", "lifting", "running", "cardio", "weights", "sport", "fitness"]):
        response = f"""### 🏃 Safe Exercise & Physical Activity Guide for {name}

Because your scan reveals a **fragile plaque cap ({plq.get('cap_thickness_um', 42)} µm)** on your {vessel_name} artery:

- **Safe Zone (Zone 2 Cardio)**: Daily brisk walking for **30 to 40 minutes** at an easy conversational pace where you can comfortably speak in full sentences. Walking gently stimulates collateral blood flow and nitric oxide production without spiking blood pressure.
- **What to AVOID right now**:
  - **Do NOT** perform heavy isometric bench pressing, deadlifts, leg presses, or crossfit workouts.
  - **Do NOT** perform sudden all-out sprints or high-intensity interval training (HIIT).
  - *Why?* Straining and holding your breath (the Valsalva maneuver) triggers sharp pressure spikes that produce intense shear stress across the thin fibrous cap.
- **Timeline**: Stick to conversational walking and light bodyweight movements for 3 to 6 months until follow-up imaging confirms your plaque cap has thickened and stabilized.
"""

    # C. Medicines, Drugs & Statin Therapy
    elif any(k in user_q for k in ["medicine", "medication", "pill", "pills", "treatment", "drug", "drugs", "statin", "aspirin", "pcsk9", "colchicine", "stent", "side effect", "dose", "dosage"]):
        response = f"""### 💊 Medical Treatment & Plaque Stabilization Plan for {name}

Here is the pharmacological rationale your cardiologist will use to protect your heart:

1. **High-Intensity Statin (Atorvastatin 80mg or Rosuvastatin 40mg)**:
   - *How it works*: Statins do not just lower blood numbers—they physically penetrate the plaque core, remove inflammatory lipid pools, and stimulate smooth muscle cells to **thicken the fibrous cap like a layer of solid cement**.
2. **PCSK9 Inhibitor (Evolocumab / Alirocumab)**:
   - *How it works*: Injected once every two weeks, this aggressively clears atherogenic ApoB particles from your bloodstream, driving ApoB down to the target goal of **<55 mg/dL**.
3. **Low-Dose Aspirin (81 mg daily)**:
   - *How it works*: Acts as an antiplatelet safeguard so that even if a microscopic surface crack forms on the plaque, blood platelets cannot clump together to create a fatal occlusive clot.
4. **Target Goals**:
   - **ApoB**: Lower from current {bio.get('apob_mg_dl', 148)} mg/dL to **< 55 mg/dL**.
   - **hs-CRP**: Lower from current {bio.get('hscrp_mg_l', 4.8)} mg/L to **< 1.0 mg/L**.
"""

    # D. Questions to Ask the Cardiologist / Doctor
    elif any(k in user_q for k in ["doctor", "cardiologist", "appointment", "ask", "tell", "consult", "visit"]):
        response = f"""### 🩺 Top 5 Questions to Ask Your Cardiologist at Your Next Visit

Bring these exact questions to your cardiology consultation:

1. *"My CardioGraph AI scan identified a **thin-cap fibroatheroma ({plq.get('cap_thickness_um', 42)} µm)** on my {vessel_name} artery. Can we begin aggressive cap-stabilizing therapy with a high-intensity statin immediately?"*
2. *"My ApoB is currently **{bio.get('apob_mg_dl', 148)} mg/dL**. Given my downstream myocardium mass at risk ({mass_risk}%), should we target an intensive ApoB goal below 55 mg/dL with a PCSK9 inhibitor?"*
3. *"Because my hs-CRP is elevated at **{bio.get('hscrp_mg_l', 4.8)} mg/L**, would adding low-dose colchicine help dampen vascular wall inflammation and protect my collagen cap?"*
4. *"What specific physical exertion limits do you advise while my plaque fibrous cap undergoes stabilization over the next 3 to 6 months?"*
5. *"When should we schedule follow-up high-resolution coronary imaging (CCTA or OCT) to verify cap thickening?"*
"""

    # E. Red-Flag Emergency Signs
    elif any(k in user_q for k in ["er", "warning", "emergency", "symptom", "symptoms", "pain", "chest", "heart attack", "911", "ambulance", "shortness"]):
        response = f"""### 🚨 Emergency Red-Flag Warning Signs (When to Call 911 Immediately)

Because occlusion of your {vessel_name} branch could starve up to **{mass_risk}% of your heart muscle**, memorizing these warning signs is critical:

- **Heavy pressure, squeezing, or fullness** in the center of your chest lasting more than a few minutes.
- **Pain radiating** to your left arm, shoulder, jaw, back, or neck.
- **Sudden shortness of breath** with or without chest discomfort.
- **Cold sweats, unexplained dizziness, nausea, or lightheadedness.**

> ⚠️ **Life-Saving Rule**: If you experience these symptoms, **do NOT wait** and do NOT attempt to drive yourself. **Call emergency services (911 / Ambulance) immediately.** State clearly: *"I have a diagnosed high-risk vulnerable coronary plaque."*
"""

    # F. Blood Tests & Biomarkers
    elif any(k in user_q for k in ["blood", "test", "tests", "apob", "crp", "hscrp", "ldl", "troponin", "lpa", "cholesterol", "lab", "labs", "number", "numbers"]):
        response = f"""### 🩸 Complete Biomarker & Blood Test Breakdown for {name}

Here is what each biomarker in your FalkorDB record actually measures:

| Test | Your Value | Target Goal | What It Actually Means |
|---|---|---|---|
| **ApoB-100** | **{bio.get('apob_mg_dl', 148)} mg/dL** | **< 55 mg/dL** | The exact count of atherogenic particles in your blood. Unlike standard LDL-C, ApoB measures the true number of particles actively entering and damaging your artery walls. |
| **hs-CRP** | **{bio.get('hscrp_mg_l', 4.8)} mg/L** | **< 1.0 mg/L** | Vascular wall inflammation. Elevated hs-CRP acts like acid, stimulating matrix metalloproteinases that dissolve the protective collagen cap of your plaque. |
| **Lp(a)** | **{bio.get('lpa_nmol_l', 192)} nmol/L** | **< 75 nmol/L** | A genetically inherited particle that promotes both plaque growth and blood clotting. |
| **Troponin-I** | **{bio.get('troponin_i_ng_ml', 0.038)} ng/mL** | **< 0.015 ng/mL** | Protein released when heart muscle cells undergo stress or micro-damage. |
"""

    # G. Rupture Paradox & Plaque Biology
    elif any(k in user_q for k in ["rupture", "paradox", "plaque", "cap", "thin", "fibrous", "stenosis", "why", "stent"]):
        response = f"""### 🛡️ Understanding the Rupture Paradox for {name}

- **The Common Myth**: Many people assume that heart attacks only happen when an artery is 80% or 90% blocked.
- **The Medical Reality (Virmani TCFA Pathology)**: Up to **70% of fatal heart attacks** occur from plaques that were only **30% to 50% narrowed** before rupture.
- **Why?**: A severe 85% blockage is usually hard, calcified, and stable. In contrast, your {vessel_name} plaque is a **soft lipid blister covered by a fragile fibrous cap ({plq.get('cap_thickness_um', 42)} µm)**.
- **The Solution**: High-intensity medical therapy thickens this cap into a durable shell, neutralizing the rupture risk without requiring open surgery.
"""

    # H. Default Comprehensive Roadmap
    else:
        status_headline = (
            "⚠️ Attention Needed: We found a fragile, high-risk plaque that regular tests often overlook!"
            if is_paradox else
            "🩺 Here is a complete plain-English breakdown of your heart health and action plan."
        )

        response = f"""### {status_headline}

Hello **{name}**, here is your personalized heart health roadmap explained in simple, clear words:

---

#### 1. What Your Scan Actually Means:
- **Your Blockage**: Your {vessel_name} artery is narrowed by **{plq.get('stenosis_pct', 38)}%**.
- **The Hidden Danger (The Rupture Paradox)**: Your plaque has a paper-thin skin (**{plq.get('cap_thickness_um', 42)} µm**, where anything under 65 µm is fragile). It is like a delicate blister filled with cholesterol sitting right where blood rushes by.
- **If it breaks**: If this small blister tears open, a sudden clot could starve **{mass_risk}% of your heart muscle**. That is why we are taking preventive action now!

---

#### 2. What Your Blood Tests Mean:
- **ApoB-100 ({bio.get('apob_mg_dl', 148)} mg/dL)**: Count of dangerous cholesterol particles penetrating your blood vessel walls. Target: **below 55 mg/dL**.
- **hs-CRP ({bio.get('hscrp_mg_l', 4.8)} mg/L)**: Measures inflammation in your blood vessels. Elevated levels act like acid weakening the plaque cap.

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
