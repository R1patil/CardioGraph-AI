/**
 * CardioGraph AI — Clinical & Context Frontend Application
 * ========================================================
 * Interactive 17-Segment Coronary Vascular Flow Tree, Acute Plaque Rupture
 * Simulation, 17-Segment AHA Myocardial Perfusion Territory Map,
 * Demis Hassabis ApoB100 Biomarker Matrix, and LangGraph Care Agent.
 */

const API_BASE = '';

// ============================================================================
// STATE
// ============================================================================

let allPatients = [];
let selectedPatientId = null;
let selectedPatientData = null;
let topologyData = null;
let currentMode = 'patient';         // 'patient' or 'doctor'
let activeTab = 'tab-vascular';
let isSplitView = false;
let isChatLoading = false;
let simulationState = 'normal';      // 'normal' or 'ruptured' or 'stented'
let stemiInterval = null;
let stemiSecondsLeft = 1200;         // 20 minutes countdown
let activeTreeFilter = 'all';        // 'all', 'LCA', 'LCX', 'RCA'
let selectedSegmentId = null;

// ============================================================================
// SVG VIEWBOX ZOOM & PAN ENGINE STATE
// ============================================================================

const BASE_VIEWBOX = { x: 0, y: 0, w: 960, h: 540 };
let currentViewBox = { ...BASE_VIEWBOX };
let targetViewBox = { ...BASE_VIEWBOX };
let zoomAnimId = null;
let isPanning = false;
let panStart = { x: 0, y: 0 };
let panViewBoxStart = { ...BASE_VIEWBOX };

// ============================================================================
// CLINICAL ANATOMICAL KNOWLEDGE MAP (PLAIN-ENGLISH FOR PATIENTS & DOCTORS)
// ============================================================================

const ANATOMY_KNOWLEDGE = {
  'SEG_LM': {
    title: 'Left Main Trunk (LM)',
    system: 'Left Coronary System',
    role: 'The master trunk arising directly from the aorta. Feeds over 70% of the entire left ventricle pumping chamber.',
    danger: 'Destabilization here is catastrophic: both the LAD and LCx branches downstream immediately lose perfusion, causing massive cardiogenic shock.',
    benefit: 'CardioGraph AI continuously tracks carinal bifurcation turbulence at the LM fork to detect early shear destabilization before occlusion can happen.'
  },
  'SEG_LAD_PROX': {
    title: 'Proximal Left Anterior Descending (LAD)',
    system: 'Left Anterior System',
    role: 'Known clinically as the "Widow Maker". Supplies the entire front wall, anterior septum, and apex of the heart.',
    danger: 'Marcus Vance has a soft lipid plaque narrowing the vessel by only 38%. Standard angiograms overlook this as "mild". But its cap is paper-thin (42 µm)! Located near high bifurcation turbulence, this fragile cap can tear in seconds, causing sudden cardiac death.',
    benefit: 'CardioGraph AI computes the ApoB-100 hazard index and Wall Shear Stress, identifying this 38% lesion as CRITICAL and prescribing intensive plaque stabilization (targeting ApoB <55 mg/dL) to prevent rupture.'
  },
  'SEG_D1': {
    title: 'First Diagonal Branch (D1)',
    system: 'Left Anterior System',
    role: 'Branches off the LAD to supply the anterolateral wall of the left ventricle.',
    danger: 'Occlusion causes regional anterolateral ischemia and impairs heart muscle contractility during exertion.',
    benefit: 'GraphBLAS reachability maps the exact lateral myocardial mass fed by D1 (11% of LV).'
  },
  'SEG_LAD_MID': {
    title: 'Mid Left Anterior Descending (LAD)',
    system: 'Left Anterior System',
    role: 'Runs down the anterior interventricular groove, supplying the deep muscular septum.',
    danger: 'Stenosis here restricts flow to septal branches, leading to angina pectoris and heart failure on exertion.',
    benefit: 'Revascularization / intensive statin therapy restores laminar blood flow and prevents distal ischemic necrosis.'
  },
  'SEG_D2': {
    title: 'Second Diagonal Branch (D2)',
    system: 'Left Anterior System',
    role: 'Distal anterolateral branch supporting the mid-to-apical lateral wall of the left ventricle.',
    danger: 'Localized ischemia leading to decreased systolic wall thickening.',
    benefit: 'Evaluated automatically in the 17-segment AHA perfusion model.'
  },
  'SEG_S1': {
    title: 'First Septal Perforator (S1)',
    system: 'Left Anterior System',
    role: 'Dives deep into the muscular ventricular septum, feeding the electrical conduction bundle of His.',
    danger: 'Ischemia here triggers cardiac conduction blocks and bundle branch delays.',
    benefit: 'Monitored as a critical flow edge in the FalkorDB vascular network.'
  },
  'SEG_LAD_DIST': {
    title: 'Distal LAD (Apex)',
    system: 'Left Anterior System',
    role: 'Wraps around the apex (the bottom pumping tip of the heart) to supply apical myocardium.',
    danger: 'Distal embolization or thrombosis causes apical ballooning and loss of apex ejection power.',
    benefit: 'Early detection ensures complete distal perfusion preservation.'
  },
  'SEG_LCX_PROX': {
    title: 'Proximal Circumflex (LCx)',
    system: 'Circumflex System',
    role: 'Courses along the left atrioventricular groove like a belt, supplying the left atrium and posterolateral LV wall.',
    danger: 'Plaque rupture causes acute lateral myocardial infarction (ECG leads I, aVL, V5, V6) and acute mitral valve insufficiency.',
    benefit: 'Coronary flow simulation checks bifurcation split resistance from the Left Main.'
  },
  'SEG_OM1': {
    title: 'First Obtuse Marginal (OM1)',
    system: 'Circumflex System',
    role: 'Large lateral branch supplying the high lateral left ventricular wall.',
    danger: 'High shear stress bifurcation point prone to soft plaque formation.',
    benefit: 'Evaluates lateral wall perfusion weight (11% LV mass).'
  },
  'SEG_LCX_DIST': {
    title: 'Distal Circumflex (LCx)',
    system: 'Circumflex System',
    role: 'Supplies the inferolateral wall and posterior ventricular base.',
    danger: 'Gradual stenosis causes chronic refractory angina.',
    benefit: 'Integrated with AHA segment 11 and 16 perfusion tracking.'
  },
  'SEG_OM2': {
    title: 'Second Obtuse Marginal (OM2)',
    system: 'Circumflex System',
    role: 'Supplies the mid-to-apical lateral ventricular myocardium.',
    danger: 'Lateral wall hypokinesis under physical stress.',
    benefit: 'Mapped directly in the FalkorDB directed flow graph.'
  },
  'SEG_RCA_PROX': {
    title: 'Proximal Right Coronary Artery (RCA)',
    system: 'Right Coronary System',
    role: 'Main arterial trunk for the right side and inferior (bottom) wall of the heart. Feeds the SA node (pacemaker).',
    danger: 'Acute occlusion triggers an Inferior STEMI and severe electrical heart block (sinus arrest or AV block), often requiring an emergency pacemaker.',
    benefit: 'Predicts threatened inferior wall mass and alerts cath lab teams to prepare temporary pacing.'
  },
  'SEG_RCA_MID': {
    title: 'Mid Right Coronary Artery (RCA)',
    system: 'Right Coronary System',
    role: 'Courses down the right atrioventricular sulcus, sustaining the right ventricular free wall.',
    danger: 'Right ventricular infarction with profound systemic hypotension.',
    benefit: 'Monitored for calcified vs lipid plaque phenotype transitions.'
  },
  'SEG_AM': {
    title: 'Acute Marginal Branch (AM)',
    system: 'Right Coronary System',
    role: 'Supplies the anterior right ventricular wall.',
    danger: 'Right ventricular dysfunction and elevated venous pressures.',
    benefit: 'Models Poiseuille flow splits along the right coronary bed.'
  },
  'SEG_RCA_DIST': {
    title: 'Distal Right Coronary Artery (RCA)',
    system: 'Right Coronary System',
    role: 'Reaches the crux of the heart to branch into the Posterior Descending Artery (PDA).',
    danger: 'Ischemia affecting the AV node and inferior ventricular base.',
    benefit: 'Calculates collateral flow availability from the left system.'
  },
  'SEG_PDA': {
    title: 'Posterior Descending Artery (PDA)',
    system: 'Right Coronary System',
    role: 'Feeds the inferior third of the interventricular septum and inferior wall.',
    danger: 'Inferior septal infarction leading to ventricular septal rupture.',
    benefit: 'Crucial for determining coronary dominance in treatment planning.'
  },
  'SEG_PLV': {
    title: 'Posterior Left Ventricular (PLV)',
    system: 'Right Coronary System',
    role: 'Supplies the diaphragmatic surface of the left ventricle.',
    danger: 'Posterior wall hypoperfusion and diaphragmatic chest pain.',
    benefit: 'Traces complete multi-hop reachability in FalkorDB.'
  }
};

// ============================================================================
// ANATOMICAL CANONICAL SVG COORDINATES MAP (960 x 540 viewBox)
// ============================================================================

const ARTERY_PATHS = {
  // Left Coronary System (LCA)
  'SEG_LM': {
    name: 'Left Main (LM)',
    caliber: 10,
    d: 'M 450,75 C 420,95 395,115 375,135',
    labelPos: { x: 395, y: 105 },
    vesselSystem: 'LCA',
    start: { x: 450, y: 75 },
    end: { x: 375, y: 135 }
  },
  'SEG_LAD_PROX': {
    name: 'Proximal LAD',
    caliber: 8.5,
    d: 'M 375,135 C 355,165 340,185 330,215',
    labelPos: { x: 335, y: 175 },
    vesselSystem: 'LCA',
    plaquePoint: { x: 350, y: 175 },
    start: { x: 375, y: 135 },
    end: { x: 330, y: 215 }
  },
  'SEG_D1': {
    name: 'First Diagonal (D1)',
    caliber: 5.5,
    d: 'M 330,215 C 290,235 255,255 230,280',
    labelPos: { x: 270, y: 240 },
    vesselSystem: 'LCA',
    plaquePoint: { x: 270, y: 250 },
    start: { x: 330, y: 215 },
    end: { x: 230, y: 280 }
  },
  'SEG_LAD_MID': {
    name: 'Mid LAD',
    caliber: 7,
    d: 'M 330,215 C 322,250 318,280 310,315',
    labelPos: { x: 300, y: 265 },
    vesselSystem: 'LCA',
    plaquePoint: { x: 320, y: 265 },
    start: { x: 330, y: 215 },
    end: { x: 310, y: 315 }
  },
  'SEG_D2': {
    name: 'Second Diagonal (D2)',
    caliber: 5,
    d: 'M 310,315 C 275,340 245,365 220,395',
    labelPos: { x: 255, y: 350 },
    vesselSystem: 'LCA',
    start: { x: 310, y: 315 },
    end: { x: 220, y: 395 }
  },
  'SEG_S1': {
    name: 'Septal Perforator (S1)',
    caliber: 4.5,
    d: 'M 310,315 C 330,335 348,355 365,375',
    labelPos: { x: 350, y: 340 },
    vesselSystem: 'LCA',
    start: { x: 310, y: 315 },
    end: { x: 365, y: 375 }
  },
  'SEG_LAD_DIST': {
    name: 'Distal LAD (Apex)',
    caliber: 5.5,
    d: 'M 310,315 C 302,360 300,410 320,490',
    labelPos: { x: 290, y: 420 },
    vesselSystem: 'LCA',
    plaquePoint: { x: 305, y: 400 },
    start: { x: 310, y: 315 },
    end: { x: 320, y: 490 }
  },

  // Circumflex System (LCx)
  'SEG_LCX_PROX': {
    name: 'Proximal LCx',
    caliber: 8,
    d: 'M 375,135 C 410,150 445,165 480,185',
    labelPos: { x: 430, y: 155 },
    vesselSystem: 'LCX',
    plaquePoint: { x: 430, y: 160 },
    start: { x: 375, y: 135 },
    end: { x: 480, y: 185 }
  },
  'SEG_OM1': {
    name: 'First Marginal (OM1)',
    caliber: 6,
    d: 'M 480,185 C 515,210 545,235 570,265',
    labelPos: { x: 535, y: 220 },
    vesselSystem: 'LCX',
    plaquePoint: { x: 525, y: 225 },
    start: { x: 480, y: 185 },
    end: { x: 570, y: 265 }
  },
  'SEG_LCX_DIST': {
    name: 'Distal LCx',
    caliber: 6,
    d: 'M 480,185 C 495,220 505,255 515,290',
    labelPos: { x: 485, y: 240 },
    vesselSystem: 'LCX',
    plaquePoint: { x: 500, y: 240 },
    start: { x: 480, y: 185 },
    end: { x: 515, y: 290 }
  },
  'SEG_OM2': {
    name: 'Second Marginal (OM2)',
    caliber: 5,
    d: 'M 515,290 C 538,320 558,350 575,380',
    labelPos: { x: 555, y: 330 },
    vesselSystem: 'LCX',
    start: { x: 515, y: 290 },
    end: { x: 575, y: 380 }
  },

  // Right Coronary System (RCA)
  'SEG_RCA_PROX': {
    name: 'Proximal RCA',
    caliber: 8.5,
    d: 'M 510,75 C 570,95 640,120 670,180',
    labelPos: { x: 605, y: 120 },
    vesselSystem: 'RCA',
    plaquePoint: { x: 610, y: 130 },
    start: { x: 510, y: 75 },
    end: { x: 670, y: 180 }
  },
  'SEG_RCA_MID': {
    name: 'Mid RCA',
    caliber: 7.5,
    d: 'M 670,180 C 685,225 680,270 665,315',
    labelPos: { x: 690, y: 250 },
    vesselSystem: 'RCA',
    plaquePoint: { x: 680, y: 250 },
    start: { x: 670, y: 180 },
    end: { x: 665, y: 315 }
  },
  'SEG_AM': {
    name: 'Acute Marginal (AM)',
    caliber: 5,
    d: 'M 678,250 C 710,275 735,300 755,330',
    labelPos: { x: 725, y: 285 },
    vesselSystem: 'RCA',
    start: { x: 678, y: 250 },
    end: { x: 755, y: 330 }
  },
  'SEG_RCA_DIST': {
    name: 'Distal RCA',
    caliber: 6.5,
    d: 'M 665,315 C 650,355 625,385 600,410',
    labelPos: { x: 645, y: 365 },
    vesselSystem: 'RCA',
    plaquePoint: { x: 635, y: 360 },
    start: { x: 665, y: 315 },
    end: { x: 600, y: 410 }
  },
  'SEG_PDA': {
    name: 'Posterior Descending (PDA)',
    caliber: 5.5,
    d: 'M 600,410 C 565,435 530,460 490,480',
    labelPos: { x: 545, y: 440 },
    vesselSystem: 'RCA',
    start: { x: 600, y: 410 },
    end: { x: 490, y: 480 }
  },
  'SEG_PLV': {
    name: 'Posterior Lateral (PLV)',
    caliber: 5,
    d: 'M 600,410 C 625,435 645,460 660,485',
    labelPos: { x: 640, y: 445 },
    vesselSystem: 'RCA',
    start: { x: 600, y: 410 },
    end: { x: 660, y: 485 }
  }
};

const BIFURCATION_POINTS = {
  'BIF_LM_LAD_LCX': { x: 375, y: 135, name: 'LM Carina (LAD/LCx)', angle: 78, shear: 8.2 },
  'BIF_LAD_D1': { x: 330, y: 215, name: 'LAD-D1 Bifurcation', angle: 65, shear: 6.5 },
  'BIF_LAD_MID_D2_S1': { x: 310, y: 315, name: 'LAD Mid Trifurcation', angle: 72, shear: 7.1 },
  'BIF_LCX_OM1': { x: 480, y: 185, name: 'LCx-OM1 Bifurcation', angle: 62, shear: 5.8 },
  'BIF_RCA_PDA_PLB': { x: 600, y: 410, name: 'Crux Cordis Bifurcation', angle: 85, shear: 6.0 }
};

// ============================================================================
// ELEMENTS
// ============================================================================

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const elPatientList = $('#patient-list');
const elWelcome = $('#welcome-screen');
const elDashboard = $('#patient-dashboard');
const elPatientAvatar = $('#patient-avatar');
const elPatientName = $('#patient-name');
const elPatientMeta = $('#patient-meta');
const elRiskBadge = $('#risk-badge');
const elParadoxIndicator = $('#paradox-indicator');
const elReasoningSection = $('#reasoning-section');
const elReasoningSteps = $('#reasoning-steps');
const elChatMessages = $('#chat-messages');
const elChatInput = $('#chat-input');
const elChatSendBtn = $('#chat-send-btn');
const elQuickQuestions = $('#quick-questions');
const elUploadModal = $('#upload-modal');
const elInspectorModal = $('#inspector-modal');

// SVG Elements
const elCoronarySvg = $('#coronary-tree-svg');
const elSvgPaths = $('#svg-artery-paths');
const elSvgBifurcations = $('#svg-bifurcations');
const elSvgPlaques = $('#svg-plaques');
const elSvgStentOverlay = $('#svg-stent-overlay');
const elSvgLabels = $('#svg-labels');
const elVesselInspectorCard = $('#vessel-inspector-card');
const elStemiHudBanner = $('#stemi-hud-banner');

// Modals & Zoom Elements
const elVesselZoomModal = $('#vessel-zoom-modal');
const elGuideModal = $('#guide-modal');
const elCrossSectionSvgContainer = $('#cross-section-svg-container');
const elZoomLevelBadge = $('#zoom-level-badge');

// ============================================================================
// INITIALIZATION
// ============================================================================

async function init() {
  try {
    const res = await fetch(`${API_BASE}/api/patients`);
    if (!res.ok) throw new Error('Failed to load patients');
    const data = await res.json();
    allPatients = data.patients || [];
    renderPatientList();
  } catch (err) {
    console.error('Init error:', err);
    elPatientList.innerHTML = `<p style="color:var(--text-muted);font-size:0.75rem;padding:8px;">Unable to load patients. Is the server running?</p>`;
  }

  // Check engine status
  try {
    const res = await fetch(`${API_BASE}/api/graph-status`);
    const status = await res.json();
    const dot = $('.status-dot');
    const txt = $('.status-text');
    if (status.falkordb_connected) {
      dot.style.background = 'var(--accent-emerald)';
      txt.textContent = 'FalkorDB GraphBLAS';
      txt.style.color = 'var(--accent-emerald)';
    } else {
      dot.style.background = 'var(--accent-amber)';
      txt.textContent = 'Local Graph Engine';
      txt.style.color = 'var(--accent-amber)';
    }
  } catch (e) { /* ignore */ }

  setupEventListeners();
  renderQuickPrompts();
}

// ============================================================================
// PATIENT LIST & RISK
// ============================================================================

function getRiskLevel(patient) {
  const plaques = patient.plaques || [];
  if (plaques.length === 0) return 'healthy';

  for (const p of plaques) {
    const cap = p.cap_thickness_um || 999;
    const vuln = p.vulnerability_score || 0;
    const sten = p.stenosis_pct || 0;

    if (cap < 65 && vuln >= 0.7) return 'critical';
    if (sten >= 70) return 'severe';
  }
  return 'moderate';
}

function getRiskLabel(level) {
  switch (level) {
    case 'critical': return 'Rupture Risk';
    case 'severe': return 'Severe Stenosis';
    case 'moderate': return 'Under Monitoring';
    case 'healthy': return 'Healthy';
    default: return '';
  }
}

function renderPatientList() {
  elPatientList.innerHTML = '';
  $('#cohort-count').textContent = `${allPatients.length} Profiles`;

  allPatients.forEach(p => {
    const risk = getRiskLevel(p);
    const item = document.createElement('div');
    item.className = `patient-item${p.id === selectedPatientId ? ' active' : ''}`;
    item.dataset.pid = p.id;
    item.innerHTML = `
      <div class="patient-dot ${risk}"></div>
      <div class="patient-item-info">
        <div class="patient-item-name">${p.name || p.id}</div>
        <div class="patient-item-tag">${p.id} · ${getRiskLabel(risk)}</div>
      </div>
    `;
    item.addEventListener('click', () => selectPatient(p.id));
    elPatientList.appendChild(item);
  });
}

// ============================================================================
// SELECT PATIENT & RENDER DASHBOARD
// ============================================================================

async function selectPatient(pid) {
  selectedPatientId = pid;
  selectedSegmentId = null;
  resetSimulationState();
  resetCanvasZoom();

  // Ensure clean single-view mode (disable split-view by default to prevent headache)
  isSplitView = false;
  $('#main-content').classList.remove('split-active');
  $('#btn-split-toggle').classList.remove('active');
  switchTab('tab-vascular');

  // Highlight in sidebar
  $$('.patient-item').forEach(el => {
    el.classList.toggle('active', el.dataset.pid === pid);
  });

  // Switch to dashboard
  elWelcome.style.display = 'none';
  elDashboard.style.display = 'flex';

  try {
    const res = await fetch(`${API_BASE}/api/patient/${pid}`);
    if (!res.ok) throw new Error('Failed to load patient graph data');
    const data = await res.json();
    selectedPatientData = data.patient;
    topologyData = data.topology;
  } catch (e) {
    console.error('Error loading patient data:', e);
    selectedPatientData = allPatients.find(p => p.id === pid);
  }

  renderPatientHeader();
  renderRiskCards();
  renderCoronaryTree();
  renderMyocardiumZones();
  renderBiomarkers();
  loadCypherQueries(pid);
  resetChat();
}


function renderPatientHeader() {
  const p = selectedPatientData;
  if (!p) return;

  const initials = (p.name || 'P').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
  elPatientAvatar.textContent = initials;
  elPatientName.textContent = p.name || p.id;

  const parts = [];
  if (p.age) parts.push(`${p.age}yo`);
  if (p.gender) parts.push(p.gender);
  if (p.smoker) parts.push('Smoker');
  if (p.diabetes) parts.push('Diabetic');
  if (p.hypertension) parts.push('Hypertensive');
  elPatientMeta.textContent = parts.join(' · ') || p.id;

  const risk = getRiskLevel(p);
  elRiskBadge.textContent = getRiskLabel(risk);
  elRiskBadge.className = `risk-badge ${risk}`;

  // Check Rupture Paradox (<50% stenosis with high vulnerability)
  const primaryPlaque = (p.plaques || [])[0] || {};
  const isParadox = (primaryPlaque.stenosis_pct < 50 && (primaryPlaque.cap_thickness_um < 65 || primaryPlaque.vulnerability_score >= 0.7));
  if (isParadox) {
    elParadoxIndicator.style.display = 'inline-flex';
  } else {
    elParadoxIndicator.style.display = 'none';
  }
}

function renderRiskCards() {
  const p = selectedPatientData;
  if (!p) return;

  const primaryPlaque = (p.plaques || [])[0] || {};
  const blast = primaryPlaque.downstream_blast_radius || {};

  // Stenosis
  const stenPct = primaryPlaque.stenosis_pct ?? '—';
  $('#val-stenosis').textContent = typeof stenPct === 'number' ? `${stenPct}%` : stenPct;
  $('#detail-stenosis').textContent = typeof stenPct === 'number'
    ? (stenPct >= 70 ? 'Severe obstructive stenosis' : stenPct >= 50 ? 'Moderate narrowing' : 'Visually non-obstructive (<50%)')
    : 'No plaque detected';

  // Cap Thickness
  const cap = primaryPlaque.cap_thickness_um ?? '—';
  $('#val-cap').textContent = typeof cap === 'number' ? `${cap} µm` : cap;
  $('#detail-cap').textContent = typeof cap === 'number'
    ? (cap < 65 ? '⚠️ Fragile TCFA Cap (<65 µm)' : 'Stable fibrous cap')
    : 'Normal intact intima';

  // Vulnerability
  const vuln = primaryPlaque.vulnerability_score ?? '—';
  $('#val-vuln').textContent = typeof vuln === 'number' ? `${(vuln * 100).toFixed(0)}%` : vuln;
  $('#detail-vuln').textContent = typeof vuln === 'number'
    ? (vuln >= 0.7 ? 'CRITICAL rupture probability' : vuln >= 0.4 ? 'Elevated vulnerability' : 'Stable quiescent plaque')
    : 'Minimal baseline risk';

  // Blast Radius
  const blastPct = blast.total_myocardium_mass_at_risk_pct ?? '—';
  $('#val-blast').textContent = typeof blastPct === 'number' ? `${blastPct.toFixed(0)}%` : blastPct;
  $('#detail-blast').textContent = typeof blastPct === 'number'
    ? `${blastPct.toFixed(0)}% Left Ventricular Mass at risk`
    : '0% territory threatened';
}

// ============================================================================
// INTERACTIVE CORONARY VASCULAR NETWORK (SVG CANVAS)
// ============================================================================

function renderCoronaryTree() {
  if (!elSvgPaths) return;

  elSvgPaths.innerHTML = '';
  elSvgBifurcations.innerHTML = '';
  elSvgPlaques.innerHTML = '';
  elSvgLabels.innerHTML = '';

  const p = selectedPatientData;
  const primaryPlaque = (p?.plaques || [])[0];
  const affectedSegmentId = primaryPlaque?.artery_segment_id;

  // Render Arterial Segments
  Object.keys(ARTERY_PATHS).forEach(segId => {
    const seg = ARTERY_PATHS[segId];

    // Filter check
    if (activeTreeFilter !== 'all' && seg.vesselSystem !== activeTreeFilter) {
      return;
    }

    const isSelected = (selectedSegmentId === segId);
    const isPlaqueHere = (affectedSegmentId === segId);
    
    // Check if this vessel is downstream from ruptured plaque
    let isDownstreamIschemic = false;
    let isOccludedSite = false;

    if (simulationState === 'ruptured') {
      const blast = primaryPlaque?.downstream_blast_radius || {};
      const downstreamSegs = blast.downstream_segments || [affectedSegmentId];
      if (segId === affectedSegmentId) {
        isOccludedSite = true;
      } else if (downstreamSegs.includes(segId)) {
        isDownstreamIschemic = true;
      }
    }

    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('class', `artery-vessel-group${isSelected ? ' selected' : ''}${isOccludedSite ? ' occluded-vessel' : ''}${isDownstreamIschemic ? ' ischemic-blast-vessel' : ''}`);
    group.dataset.segmentId = segId;

    // Background thick vessel lumen
    const pathBg = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathBg.setAttribute('d', seg.d);
    pathBg.setAttribute('class', 'artery-vessel-bg');
    pathBg.setAttribute('stroke-width', seg.caliber);
    pathBg.setAttribute('fill', 'none');
    
    if (simulationState === 'ruptured' && (isOccludedSite || isDownstreamIschemic)) {
      pathBg.setAttribute('stroke', '#475569');
    } else {
      pathBg.setAttribute('stroke', 'url(#grad-artery-normal)');
    }

    // Foreground animated blood flow stroke
    const pathFlow = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathFlow.setAttribute('d', seg.d);
    pathFlow.setAttribute('class', 'artery-path-flow');
    pathFlow.setAttribute('stroke-width', Math.max(2, seg.caliber - 3));
    pathFlow.setAttribute('stroke', '#ffffff');
    pathFlow.setAttribute('fill', 'none');
    pathFlow.setAttribute('opacity', simulationState === 'ruptured' && (isOccludedSite || isDownstreamIschemic) ? '0' : '0.6');

    group.appendChild(pathBg);
    group.appendChild(pathFlow);

    // Click handler for segment inspection & smooth zoom
    group.addEventListener('click', (e) => {
      e.stopPropagation();
      openVesselZoomModal(segId);
      focusOnSegment(segId);
    });

    elSvgPaths.appendChild(group);

    // Anatomical Text Label
    if (seg.labelPos) {
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', seg.labelPos.x);
      text.setAttribute('y', seg.labelPos.y);
      text.setAttribute('font-size', '9');
      text.setAttribute('font-weight', '600');
      text.setAttribute('fill', isDownstreamIschemic ? '#f87171' : '#94a3b8');
      text.setAttribute('text-anchor', 'middle');
      text.textContent = seg.name.split(' (')[0];
      elSvgLabels.appendChild(text);
    }
  });

  // Render Bifurcation Junction Nodes
  Object.keys(BIFURCATION_POINTS).forEach(bifId => {
    const bif = BIFURCATION_POINTS[bifId];
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', bif.x);
    circle.setAttribute('cy', bif.y);
    circle.setAttribute('r', '5.5');
    circle.setAttribute('class', 'bifurcation-marker-node');
    circle.setAttribute('data-bifId', bifId);
    circle.setAttribute('title', `${bif.name} (Shear Index: ${bif.shear}, Angle: ${bif.angle}°) - Click to Zoom & Inspect`);

    circle.addEventListener('click', (e) => {
      e.stopPropagation();
      openBifurcationZoomModal(bifId);
      focusOnPoint(bif.x, bif.y, 280);
    });

    elSvgBifurcations.appendChild(circle);
  });

  // Render Plaque Marker (if present)
  if (primaryPlaque && affectedSegmentId && ARTERY_PATHS[affectedSegmentId]) {
    const seg = ARTERY_PATHS[affectedSegmentId];
    const pt = seg.plaquePoint || { x: (seg.start.x + seg.end.x) / 2, y: (seg.start.y + seg.end.y) / 2 };

    const plqGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    plqGroup.setAttribute('class', 'plaque-marker-group');
    plqGroup.setAttribute('transform', `translate(${pt.x}, ${pt.y})`);

    // Pulsing danger ring if TCFA
    const isTcfa = (primaryPlaque.cap_thickness_um < 65);
    if (isTcfa) {
      const pulseRing = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      pulseRing.setAttribute('class', 'plaque-pulse-ring');
      pulseRing.setAttribute('cx', '0');
      pulseRing.setAttribute('cy', '0');
      pulseRing.setAttribute('r', '8');
      plqGroup.appendChild(pulseRing);
    }

    // Core Plaque Dot
    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('class', 'plaque-core-dot');
    dot.setAttribute('cx', '0');
    dot.setAttribute('cy', '0');
    dot.setAttribute('r', isTcfa ? '6.5' : '5');
    plqGroup.appendChild(dot);

    // Stenosis / TCFA Badge text
    const badge = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    badge.setAttribute('x', '10');
    badge.setAttribute('y', '-6');
    badge.setAttribute('font-size', '9.5');
    badge.setAttribute('font-weight', '800');
    badge.setAttribute('fill', '#f43f5e');
    badge.textContent = `${primaryPlaque.stenosis_pct}% ${isTcfa ? 'TCFA' : 'Plaque'} (Zoom)`;
    plqGroup.appendChild(badge);

    // Click handler to zoom in on plaque and open cross-section picture
    plqGroup.addEventListener('click', (e) => {
      e.stopPropagation();
      openVesselZoomModal(affectedSegmentId, true);
      focusOnPlaque();
    });

    elSvgPlaques.appendChild(plqGroup);
  }
}


// ============================================================================
// SVG VIEWBOX ZOOM, PAN & FOCUS ENGINE
// ============================================================================

function updateViewBoxDisplay() {
  if (!elCoronarySvg) return;
  elCoronarySvg.setAttribute('viewBox', `${currentViewBox.x} ${currentViewBox.y} ${currentViewBox.w} ${currentViewBox.h}`);
  if (elZoomLevelBadge) {
    const pct = Math.round((BASE_VIEWBOX.w / currentViewBox.w) * 100);
    elZoomLevelBadge.textContent = `${pct}%`;
  }
}

function animateViewBoxTo(target, duration = 350) {
  if (zoomAnimId) cancelAnimationFrame(zoomAnimId);
  const startX = currentViewBox.x;
  const startY = currentViewBox.y;
  const startW = currentViewBox.w;
  const startH = currentViewBox.h;
  const startTime = performance.now();

  function step(now) {
    const elapsed = now - startTime;
    const progress = Math.min(1, elapsed / duration);
    const ease = 1 - Math.pow(1 - progress, 3); // easeOutCubic

    currentViewBox.x = startX + (target.x - startX) * ease;
    currentViewBox.y = startY + (target.y - startY) * ease;
    currentViewBox.w = startW + (target.w - startW) * ease;
    currentViewBox.h = startH + (target.h - startH) * ease;

    updateViewBoxDisplay();

    if (progress < 1) {
      zoomAnimId = requestAnimationFrame(step);
    } else {
      currentViewBox = { ...target };
      updateViewBoxDisplay();
      zoomAnimId = null;
    }
  }

  zoomAnimId = requestAnimationFrame(step);
}

function zoomCanvas(factor, centerPoint = null) {
  // Clamp width between 180 (5.3x zoom) and 960 (1x zoom)
  const newW = Math.max(180, Math.min(BASE_VIEWBOX.w, currentViewBox.w * factor));
  const newH = newW * (BASE_VIEWBOX.h / BASE_VIEWBOX.w);

  const cx = centerPoint ? centerPoint.x : (currentViewBox.x + currentViewBox.w / 2);
  const cy = centerPoint ? centerPoint.y : (currentViewBox.y + currentViewBox.h / 2);

  const newX = Math.max(0, Math.min(BASE_VIEWBOX.w - newW, cx - newW / 2));
  const newY = Math.max(0, Math.min(BASE_VIEWBOX.h - newH, cy - newH / 2));

  animateViewBoxTo({ x: newX, y: newY, w: newW, h: newH });
}

function resetCanvasZoom() {
  selectedSegmentId = null;
  removeFocusRing();
  animateViewBoxTo(BASE_VIEWBOX, 400);
  renderCoronaryTree();
}

function removeFocusRing() {
  const existing = document.getElementById('svg-focus-ring');
  if (existing) existing.remove();
}

function addFocusRing(cx, cy, radius = 28) {
  removeFocusRing();
  if (!elSvgPaths) return;
  const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  ring.setAttribute('id', 'svg-focus-ring');
  ring.setAttribute('class', 'focus-highlight-ring');
  ring.setAttribute('cx', cx);
  ring.setAttribute('cy', cy);
  ring.setAttribute('r', radius);
  elSvgPaths.appendChild(ring);
}

function focusOnPoint(cx, cy, targetW = 340) {
  const targetH = targetW * (BASE_VIEWBOX.h / BASE_VIEWBOX.w);
  const targetX = Math.max(0, Math.min(BASE_VIEWBOX.w - targetW, cx - targetW / 2));
  const targetY = Math.max(0, Math.min(BASE_VIEWBOX.h - targetH, cy - targetH / 2));
  addFocusRing(cx, cy);
  animateViewBoxTo({ x: targetX, y: targetY, w: targetW, h: targetH });
}

function focusOnSegment(segId) {
  selectedSegmentId = segId;
  const seg = ARTERY_PATHS[segId];
  if (!seg) return;
  const cx = (seg.start.x + seg.end.x) / 2;
  const cy = (seg.start.y + seg.end.y) / 2;
  focusOnPoint(cx, cy, 320);
  renderCoronaryTree();
}

function focusOnPlaque() {
  const p = selectedPatientData;
  const plq = (p?.plaques || [])[0];
  if (!plq || !ARTERY_PATHS[plq.artery_segment_id]) {
    alert('No plaque found on this patient.');
    return;
  }
  const seg = ARTERY_PATHS[plq.artery_segment_id];
  const pt = seg.plaquePoint || { x: (seg.start.x + seg.end.x) / 2, y: (seg.start.y + seg.end.y) / 2 };
  selectedSegmentId = plq.artery_segment_id;
  focusOnPoint(pt.x, pt.y, 250);
  renderCoronaryTree();
}

// ============================================================================
// MICROSCOPIC ARTERIAL CROSS-SECTION PICTURE RENDERER
// ============================================================================

function renderArterialCrossSection(container, opts) {
  if (!container) return;
  const {
    stenosis = 38,
    capThickness = 42,
    isTcfa = true,
    isRuptured = false,
    isStented = false,
    vesselName = 'Proximal LAD'
  } = opts;

  const W = 320, H = 210;
  const cx = 160, cy = 105;
  const outerR = 80;
  const innerR = 68;

  // Lumen opening radius shrinks as stenosis increases
  const lumenR = Math.max(18, Math.round(62 * (1 - (stenosis / 100) * 0.65)));
  const lipidCoreOffset = Math.round(stenosis * 0.42);

  const capColor = isTcfa ? '#f43f5e' : '#10b981';
  const capStrokeWidth = isTcfa ? 3.5 : 7;

  const svgHtml = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" style="overflow:visible;">
      <defs>
        <radialGradient id="grad-cs-wall" cx="50%" cy="50%" r="50%">
          <stop offset="60%" stop-color="#1e293b" />
          <stop offset="90%" stop-color="#334155" />
          <stop offset="100%" stop-color="#475569" />
        </radialGradient>

        <linearGradient id="grad-lipid-pool" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#f59e0b" />
          <stop offset="60%" stop-color="#d97706" />
          <stop offset="100%" stop-color="#92400e" />
        </linearGradient>

        <radialGradient id="grad-lumen-blood" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.9" />
          <stop offset="65%" stop-color="#0284c7" stop-opacity="0.75" />
          <stop offset="100%" stop-color="#0369a1" stop-opacity="0.45" />
        </radialGradient>

        <radialGradient id="grad-thrombus" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#881337" />
          <stop offset="50%" stop-color="#991b1b" />
          <stop offset="100%" stop-color="#450a0a" />
        </radialGradient>

        <filter id="cs-glow-danger" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <!-- Outer Muscular Artery Wall (Adventitia & Media) -->
      <circle cx="${cx}" cy="${cy}" r="${outerR}" fill="url(#grad-cs-wall)" stroke="#64748b" stroke-width="3" />
      <circle cx="${cx}" cy="${cy}" r="${innerR}" fill="#0f172a" stroke="#475569" stroke-width="1.5" stroke-dasharray="4 3" />

      <!-- Soft Lipid Plaque Core (Atheroma) -->
      ${stenosis > 0 ? `
        <ellipse cx="${cx - lipidCoreOffset / 2}" cy="${cy}" rx="${lipidCoreOffset + 22}" ry="${innerR - 6}"
                 fill="url(#grad-lipid-pool)" opacity="0.95" />
        <circle cx="${cx - lipidCoreOffset / 2 - 8}" cy="${cy - 12}" r="5" fill="#fef08a" opacity="0.6" />
        <circle cx="${cx - lipidCoreOffset / 2 + 10}" cy="${cy + 15}" r="6" fill="#fef08a" opacity="0.6" />
        <circle cx="${cx - lipidCoreOffset / 2 - 14}" cy="${cy + 8}" r="4" fill="#fef08a" opacity="0.7" />
      ` : ''}

      <!-- Blood Lumen Channel -->
      <circle cx="${cx + (stenosis > 0 ? lipidCoreOffset * 0.45 : 0)}" cy="${cy}" r="${lumenR}"
              fill="${isRuptured ? 'url(#grad-thrombus)' : 'url(#grad-lumen-blood)'}"
              stroke="${isRuptured ? '#ef4444' : '#38bdf8'}" stroke-width="2" />

      <!-- Fibrous Cap Barrier -->
      ${stenosis > 0 && !isStented ? `
        <path d="M ${cx + lipidCoreOffset * 0.45 - lumenR},${cy - lumenR * 0.8}
                 C ${cx - lipidCoreOffset * 0.2},${cy} ${cx - lipidCoreOffset * 0.2},${cy} ${cx + lipidCoreOffset * 0.45 - lumenR},${cy + lumenR * 0.8}"
              stroke="${capColor}" stroke-width="${capStrokeWidth}" fill="none"
              filter="${isTcfa ? 'url(#cs-glow-danger)' : 'none'}" />
      ` : ''}

      <!-- Metallic Stent Struts (if Stented) -->
      ${isStented ? `
        <g stroke="#10b981" stroke-width="2.5" fill="none">
          <circle cx="${cx}" cy="${cy}" r="${innerR - 2}" stroke-dasharray="8 8" />
          <circle cx="${cx}" cy="${cy}" r="${innerR - 8}" stroke="#34d399" stroke-dasharray="6 6" />
        </g>
      ` : ''}

      <!-- Central Text Status -->
      ${isRuptured ? `
        <text x="${cx}" y="${cy + 4}" text-anchor="middle" fill="#fecaca" font-size="10.5" font-weight="900">CLOT (0 FLOW)</text>
      ` : `
        <text x="${cx + (stenosis > 0 ? lipidCoreOffset * 0.45 : 0)}" y="${cy + 4}" text-anchor="middle" fill="#ffffff" font-size="10" font-weight="800">
          ${isStented ? '100% TIMI-3' : `${(100 - stenosis)}% OPEN`}
        </text>
      `}

      <!-- Anatomical Callout Labels -->
      <line x1="${cx - outerR}" y1="${cy - 35}" x2="${cx - outerR - 12}" y2="${cy - 35}" stroke="#94a3b8" stroke-width="1" />
      <text x="${cx - outerR - 16}" y="${cy - 32}" fill="#94a3b8" font-size="9" text-anchor="end" font-weight="600">Artery Wall</text>

      ${stenosis > 0 ? `
        <line x1="${cx - 15}" y1="${cy + 25}" x2="${cx - 25}" y2="${cy + 58}" stroke="#f59e0b" stroke-width="1.2" />
        <text x="${cx - 28}" y="${cy + 66}" fill="#f59e0b" font-size="9" text-anchor="end" font-weight="700">Lipid Core (${stenosis}%)</text>

        <line x1="${cx + lipidCoreOffset * 0.45 - lumenR}" y1="${cy}" x2="${cx + 10}" y2="${cy - 55}" stroke="${capColor}" stroke-width="1.2" />
        <text x="${cx + 12}" y="${cy - 60}" fill="${capColor}" font-size="9" font-weight="800">${isTcfa ? `TCFA Cap (${capThickness}µm)` : `Thick Cap (${capThickness}µm)`}</text>
      ` : ''}
    </svg>
  `;

  container.innerHTML = svgHtml;
}

// ============================================================================
// ANATOMICAL VESSEL ZOOM MODAL CONTROLLER
// ============================================================================

function openVesselZoomModal(segId, isPlaqueClick = false) {
  selectedSegmentId = segId;
  const segMeta = ARTERY_PATHS[segId] || { name: segId, vesselSystem: 'Coronary' };
  const p = selectedPatientData || {};
  const primaryPlaque = (p.plaques || [])[0] || {};
  const isTargetPlaqueVessel = (primaryPlaque.artery_segment_id === segId) || isPlaqueClick;
  const info = ANATOMY_KNOWLEDGE[segId] || {
    title: segMeta.name,
    system: segMeta.vesselSystem,
    role: `Important coronary segment supplying myocardial tissue.`,
    danger: `Plaque accumulation causes regional ischemic stress.`,
    benefit: `CardioGraph AI tracks continuous flow and predicts downstream perfusion.`
  };

  // Find canonical segment data
  const canonicalSeg = (topologyData?.artery_segments || []).find(s => s.id === segId) || {
    diameter_mm: 3.2,
    length_mm: 20.0,
    baseline_flow_ml_min: 80.0,
    resistance: 0.18,
    perfuses_zones: []
  };

  // Populate header
  $('#zoom-vessel-badge').textContent = `${segId} · ${segMeta.vesselSystem || 'LCA'}`;
  $('#zoom-vessel-title').textContent = info.title;
  $('#zoom-vessel-subtitle').textContent = info.role.split('.')[0] + '.';

  // Plaque metrics
  const sten = isTargetPlaqueVessel ? (primaryPlaque.stenosis_pct || 38) : 0;
  const cap = isTargetPlaqueVessel ? (primaryPlaque.cap_thickness_um || 42) : 180;
  const isTcfa = (cap < 65);
  const shear = isTargetPlaqueVessel ? 8.5 : 4.2;
  const blastPct = isTargetPlaqueVessel ? (primaryPlaque.downstream_blast_radius?.total_myocardium_mass_at_risk_pct || 68) : 0;

  // Cap badge
  const elCapTag = $('#zoom-cap-tag');
  if (isTargetPlaqueVessel && isTcfa) {
    elCapTag.textContent = '⚠️ THIN-CAP FIBROATHEROMA (<65 µm)';
    elCapTag.className = 'picture-status-tag danger';
  } else if (isTargetPlaqueVessel) {
    elCapTag.textContent = 'STABLE FIBROCALCIFIC PLAQUE';
    elCapTag.className = 'picture-status-tag';
  } else {
    elCapTag.textContent = 'HEALTHY OPEN ARTERY';
    elCapTag.className = 'picture-status-tag';
  }

  // Update stat cards
  $('#zm-stenosis').textContent = `${sten}%`;
  $('#zm-stenosis-sub').textContent = sten >= 70 ? 'Obstructive' : sten >= 50 ? 'Moderate' : (sten > 0 ? 'Visually "Mild"' : 'Clear Lumen');

  $('#zm-cap').textContent = isTargetPlaqueVessel ? `${cap} µm` : 'Normal';
  $('#zm-cap').className = isTcfa ? 'zm-value danger' : 'zm-value';

  $('#zm-shear').textContent = `${shear} WSS`;
  $('#zm-blast').textContent = `${blastPct}%`;

  // Draw Microscopic Cross-Section Picture
  renderArterialCrossSection(elCrossSectionSvgContainer, {
    stenosis: sten,
    capThickness: cap,
    isTcfa: isTcfa,
    isRuptured: (simulationState === 'ruptured' && isTargetPlaqueVessel),
    isStented: (simulationState === 'stented' && isTargetPlaqueVessel),
    vesselName: segMeta.name
  });

  // Populate 4 Plain-English Clinical Cards
  $('#z-text-anatomy').textContent = info.role;

  $('#z-text-findings').textContent = isTargetPlaqueVessel
    ? `CardioGraph AI detected a ${sten}% soft lipid plaque with an ultra-thin ${cap} µm fibrous cap. ApoB is ${p.biomarkers?.apob_mg_dl || 148} mg/dL, causing biochemical cap degradation.`
    : `This segment has a healthy caliber of ${canonicalSeg.diameter_mm} mm with normal baseline flow of ${canonicalSeg.baseline_flow_ml_min} mL/min and no active plaque lesions.`;

  $('#z-text-danger').textContent = isTargetPlaqueVessel
    ? (isTcfa 
      ? `THE RUPTURE PARADOX: Despite only ${sten}% blockage, the thin ${cap} µm cap cannot withstand carinal blood turbulence (${shear} WSS). If it ruptures, ${blastPct}% of your heart muscle will die in 20 minutes!` 
      : info.danger)
    : `Currently low risk. Maintaining low ApoB-100 prevents foam-cell accumulation in this segment.`;

  $('#z-text-benefit').textContent = isTargetPlaqueVessel
    ? `CardioGraph AI sounds the alarm before rupture happens! Prescribing intensive ApoB reduction (<55 mg/dL with PCSK9i + statins) thickens the fibrous cap, turning a lethal soft plaque into a safe, calcified lesion.`
    : info.benefit;

  elVesselZoomModal.style.display = 'flex';
}

function openBifurcationZoomModal(bifId) {
  const bif = BIFURCATION_POINTS[bifId];
  if (!bif) return;

  $('#zoom-vessel-badge').textContent = `BIFURCATION · CARINA`;
  $('#zoom-vessel-title').textContent = bif.name;
  $('#zoom-vessel-subtitle').textContent = `Arterial bifurcation fork with high hemodynamic shear stress.`;

  $('#zoom-cap-tag').textContent = 'HIGH TURBULENCE JUNCTION';
  $('#zoom-cap-tag').className = 'picture-status-tag';

  $('#zm-stenosis').textContent = `${bif.angle}°`;
  $('#zm-stenosis-sub').textContent = 'Branch Angle';

  $('#zm-cap').textContent = `WSS ${bif.shear}`;
  $('#zm-cap').className = 'zm-value warning';

  $('#zm-shear').textContent = `High`;
  $('#zm-blast').textContent = `Critical`;

  renderArterialCrossSection(elCrossSectionSvgContainer, {
    stenosis: 25,
    capThickness: 50,
    isTcfa: true,
    isRuptured: false,
    isStented: false,
    vesselName: bif.name
  });

  $('#z-text-anatomy').textContent = `Arterial branching carina where blood flow divides into two downstream channels. Forces fluid turbulence against the vessel wall.`;
  $('#z-text-findings').textContent = `Wall Shear Stress (WSS) index of ${bif.shear} with bifurcation angle of ${bif.angle}°. Oscillating shear stress promotes endothelial erosion.`;
  $('#z-text-danger').textContent = `Plaques located within 5 mm of this bifurcation experience 3x higher mechanical fatigue, making them prone to sudden tear and rupture.`;
  $('#z-text-benefit').textContent = `CardioGraph AI incorporates bifurcation fluid dynamics directly into the Plaque Vulnerability Index (PVI).`;

  elVesselZoomModal.style.display = 'flex';
}

function closeVesselZoomModal() {
  elVesselZoomModal.style.display = 'none';
}

function openGuideModal() {
  elGuideModal.style.display = 'flex';
}

function closeGuideModal() {
  elGuideModal.style.display = 'none';
}


// ============================================================================
// ACUTE PLAQUE RUPTURE SIMULATION (CODE STEMI)
// ============================================================================

async function simulatePlaqueRupture() {
  const p = selectedPatientData;
  if (!p) return;

  const primaryPlaque = (p.plaques || [])[0];
  if (!primaryPlaque) {
    alert('This patient has no identified plaques to simulate rupture on.');
    return;
  }

  simulationState = 'ruptured';
  stemiSecondsLeft = 1200; // 20 minutes

  // Call backend rupture simulation endpoint
  try {
    const res = await fetch(`${API_BASE}/api/simulate-rupture`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plaque_id: primaryPlaque.id })
    });
    const result = await res.json();
    populateStemiHud(result);
  } catch (err) {
    console.error('Rupture sim error:', err);
    populateStemiHudDefault(primaryPlaque);
  }

  // Show STEMI HUD Banner
  elStemiHudBanner.style.display = 'block';

  // Start live 20:00 countdown clock
  startStemiCountdown();

  // Re-render tree with ischemic black/purple styling
  renderCoronaryTree();
  updateMyocardiumThreatState(true);

  // Switch to vascular tab
  switchTab('tab-vascular');

  // Trigger conversational agent emergency alert
  if (currentMode === 'doctor') {
    await sendChatMessage('CODE STEMI ACTIVATION: Primary PCI protocol, vessel occlusion, and expected ECG leads.');
  } else {
    await sendChatMessage('What happens when my plaque ruptures and what are the emergency warning signs?');
  }
}

function populateStemiHud(data) {
  $('#stemi-occluded-vessel').textContent = data.occluded_artery_segment || 'SEG_LAD_PROX';
  const blast = data.downstream_blast_radius || {};
  $('#stemi-mass-threatened').textContent = `${blast.total_myocardium_mass_at_risk_pct || 68}% LV Mass`;
  
  const ecg = data.predicted_ecg_manifestation || ['ST-Elevation in Leads V1–V4'];
  $('#stemi-ecg-leads').textContent = Array.isArray(ecg) ? ecg[0] : ecg;
  $('#stemi-protocol').textContent = 'Primary PCI & Balloon Thrombectomy';
}

function populateStemiHudDefault(plq) {
  $('#stemi-occluded-vessel').textContent = plq.artery_segment_id;
  const blast = plq.downstream_blast_radius || {};
  $('#stemi-mass-threatened').textContent = `${blast.total_myocardium_mass_at_risk_pct || 68}% LV Mass`;
  $('#stemi-ecg-leads').textContent = 'ST-Elevation in Leads V1–V4 (Anterior STEMI)';
  $('#stemi-protocol').textContent = 'CODE STEMI Activation';
}

function startStemiCountdown() {
  if (stemiInterval) clearInterval(stemiInterval);

  stemiInterval = setInterval(() => {
    stemiSecondsLeft--;
    if (stemiSecondsLeft <= 0) {
      clearInterval(stemiInterval);
      $('#stemi-timer-clock').textContent = '00:00 (IRREVERSIBLE NECROSIS)';
      return;
    }

    const mins = Math.floor(stemiSecondsLeft / 60);
    const secs = stemiSecondsLeft % 60;
    $('#stemi-timer-clock').textContent = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }, 1000);
}

function deployStentAngioplasty() {
  simulationState = 'stented';
  if (stemiInterval) clearInterval(stemiInterval);

  // Hide STEMI HUD
  elStemiHudBanner.style.display = 'none';

  // Restore normal perfusion
  renderCoronaryTree();
  updateMyocardiumThreatState(false);

  alert('✅ Percutaneous Coronary Intervention Successful:\nDrug-Eluting Stent Deployed. TIMI-3 Perfusion Flow Restored.\nMyocardial Tissue Successfully Salvaged!');

  sendChatMessage('The patient underwent emergency primary PCI and stent placement. What is the post-procedure recovery and medication protocol?');
}

function resetSimulationState() {
  simulationState = 'normal';
  if (stemiInterval) clearInterval(stemiInterval);
  elStemiHudBanner.style.display = 'none';
  $('#stemi-timer-clock').textContent = '20:00';
  updateMyocardiumThreatState(false);
}

// ============================================================================
// 17-SEGMENT AHA MYOCARDIUM PERFUSION TERRITORY MAP
// ============================================================================

function renderMyocardiumZones() {
  const container = $('#zones-grid');
  if (!container) return;

  container.innerHTML = '';
  const zones = topologyData?.myocardium_zones || [
    { id: 'ZONE_ANT_BASAL', zone_name: 'Basal Anterior', wall: 'Anterior', mass_pct: 10.0 },
    { id: 'ZONE_ANT_MID', zone_name: 'Mid Anterior', wall: 'Anterior', mass_pct: 12.0 },
    { id: 'ZONE_SEPT_BASAL', zone_name: 'Basal Septal', wall: 'Septal', mass_pct: 10.0 },
    { id: 'ZONE_SEPT_MID', zone_name: 'Mid Septal', wall: 'Septal', mass_pct: 11.0 },
    { id: 'ZONE_APICAL', zone_name: 'Apical Territory & True Apex', wall: 'Apical', mass_pct: 14.0 },
    { id: 'ZONE_LAT_BASAL', zone_name: 'Basal Lateral', wall: 'Lateral', mass_pct: 11.0 },
    { id: 'ZONE_LAT_MID', zone_name: 'Mid Lateral', wall: 'Lateral', mass_pct: 12.0 },
    { id: 'ZONE_INF_BASAL', zone_name: 'Basal Inferior', wall: 'Inferior', mass_pct: 10.0 },
    { id: 'ZONE_INF_MID', zone_name: 'Mid Inferior', wall: 'Inferior', mass_pct: 10.0 }
  ];

  zones.forEach(zone => {
    const card = document.createElement('div');
    card.className = 'zone-card healthy';
    card.id = `zone-card-${zone.id}`;
    card.dataset.zoneId = zone.id;

    card.innerHTML = `
      <div class="zone-header">
        <span class="zone-name">${zone.zone_name}</span>
        <span class="zone-wall">${zone.wall} Wall</span>
      </div>
      <div class="zone-meta">
        <span class="zone-mass">${zone.mass_pct}% Myocardial Mass</span>
        <span class="zone-status healthy" id="status-${zone.id}">Perfused 98% O₂</span>
      </div>
    `;

    card.addEventListener('click', () => {
      alert(`Myocardium Zone: ${zone.zone_name}\nWall: ${zone.wall}\nMass Percentage: ${zone.mass_pct}%\nSupplied primarily by coronary flow tree.`);
    });

    container.appendChild(card);
  });
}

function updateMyocardiumThreatState(isThreatened) {
  const p = selectedPatientData;
  const primaryPlaque = (p?.plaques || [])[0];
  const blast = primaryPlaque?.downstream_blast_radius || {};
  const threatenedZoneIds = (blast.threatened_zones || []).map(z => z.id || z.zone_id);

  const statusTag = $('#territory-status-tag');
  const statusDot = $('.status-pulse-dot');
  const statusText = $('#territory-status-text');

  if (isThreatened && threatenedZoneIds.length > 0) {
    statusDot.className = 'status-pulse-dot danger';
    statusText.textContent = `CRITICAL: ${blast.total_myocardium_mass_at_risk_pct || 68}% LV Mass Under Ischemic Necrosis Risk`;

    threatenedZoneIds.forEach(zid => {
      const card = $(`#zone-card-${zid}`);
      if (card) {
        card.className = 'zone-card threatened';
        const st = $(`#status-${zid}`);
        if (st) {
          st.className = 'zone-status threatened';
          st.textContent = 'ISCHEMIA THREAT (0% Flow)';
        }
      }
    });
  } else {
    statusDot.className = 'status-pulse-dot healthy';
    statusText.textContent = 'All 9 Territories Perfused (100% Viable)';

    $$('.zone-card').forEach(card => {
      card.className = 'zone-card healthy';
      const zid = card.dataset.zoneId;
      const st = $(`#status-${zid}`);
      if (st) {
        st.className = 'zone-status healthy';
        st.textContent = 'Perfused 98% O₂';
      }
    });
  }
}

// ============================================================================
// BIOMARKERS PANEL (DEMIS HASSABIS ApoB & INFLAMMATION)
// ============================================================================

function renderBiomarkers() {
  const p = selectedPatientData;
  if (!p) return;

  const bio = p.biomarkers || {};

  // ApoB (Target < 70)
  const apob = bio.apob_mg_dl ?? 148.0;
  $('#val-apob').textContent = apob.toFixed(1);
  const apobDelta = apob - 70;
  $('#delta-apob').textContent = apobDelta > 0 ? `+${apobDelta.toFixed(1)} mg/dL above goal` : 'Optimal target achieved';
  $('#badge-apob').className = `bio-badge ${apob > 100 ? 'danger' : (apob > 70 ? 'warning' : 'success')}`;
  $('#badge-apob').textContent = apob > 100 ? 'CRITICAL HIGH' : (apob > 70 ? 'ELEVATED' : 'OPTIMAL');
  $('#fill-apob').style.width = `${Math.min(100, (apob / 200) * 100)}%`;

  // hs-CRP (Target < 1.0)
  const hscrp = bio.hscrp_mg_l ?? 4.8;
  $('#val-hscrp').textContent = hscrp.toFixed(1);
  const crpDelta = hscrp - 1.0;
  $('#delta-hscrp').textContent = crpDelta > 0 ? `+${crpDelta.toFixed(1)} mg/L elevated` : 'Low systemic inflammation';
  $('#badge-hscrp').className = `bio-badge ${hscrp > 3.0 ? 'danger' : (hscrp > 1.0 ? 'warning' : 'success')}`;
  $('#badge-hscrp').textContent = hscrp > 3.0 ? 'HIGH INFLAMMATION' : (hscrp > 1.0 ? 'MODERATE' : 'LOW RISK');
  $('#fill-hscrp').style.width = `${Math.min(100, (hscrp / 10) * 100)}%`;

  // Lp(a) (Target < 75)
  const lpa = bio.lpa_nmol_l ?? 192.0;
  $('#val-lpa').textContent = lpa.toFixed(1);
  const lpaDelta = lpa - 75;
  $('#delta-lpa').textContent = lpaDelta > 0 ? `+${lpaDelta.toFixed(1)} nmol/L genetic risk` : 'Normal range';
  $('#badge-lpa').className = `bio-badge ${lpa > 125 ? 'danger' : (lpa > 75 ? 'warning' : 'success')}`;
  $('#badge-lpa').textContent = lpa > 125 ? 'VERY HIGH GENETIC RISK' : (lpa > 75 ? 'BORDERLINE' : 'NORMAL');
  $('#fill-lpa').style.width = `${Math.min(100, (lpa / 300) * 100)}%`;

  // LDL-C (Target < 70)
  const ldl = bio.ldl_c_mg_dl ?? 162.0;
  $('#val-ldl').textContent = ldl.toFixed(1);
  const ldlDelta = ldl - 70;
  $('#delta-ldl').textContent = ldlDelta > 0 ? `+${ldlDelta.toFixed(1)} mg/dL above target` : 'At goal';
  $('#badge-ldl').className = `bio-badge ${ldl > 130 ? 'danger' : (ldl > 70 ? 'warning' : 'success')}`;
  $('#badge-ldl').textContent = ldl > 130 ? 'ELEVATED' : (ldl > 70 ? 'BORDERLINE' : 'OPTIMAL');
  $('#fill-ldl').style.width = `${Math.min(100, (ldl / 220) * 100)}%`;

  // Troponin-I (Target < 0.04)
  const trop = bio.troponin_i_ng_ml ?? 0.038;
  $('#val-trop').textContent = trop.toFixed(3);
  $('#delta-trop').textContent = trop < 0.04 ? 'Normal baseline (no necrosis)' : '⚠️ Acute myocyte leakage';
  $('#badge-trop').className = `bio-badge ${trop >= 0.04 ? 'danger' : 'success'}`;
  $('#badge-trop').textContent = trop >= 0.04 ? 'ELEVATED' : 'BASELINE NORMAL';
  $('#fill-trop').style.width = `${Math.min(100, (trop / 0.1) * 100)}%`;
}

// ============================================================================
// CYPHER TRAVERSAL & MCP AGENT TOOLS
// ============================================================================

async function loadCypherQueries(pid) {
  try {
    const res = await fetch(`${API_BASE}/api/cypher-queries/${pid}`);
    if (!res.ok) throw new Error('Failed to load Cypher queries');
    const data = await res.json();
    renderCypherQueries(data.queries || []);
  } catch (err) {
    console.error('Cypher query load error:', err);
  }
}

function renderCypherQueries(queries) {
  const container = $('#cypher-queries-list');
  if (!container) return;

  container.innerHTML = '';
  queries.forEach(q => {
    const card = document.createElement('div');
    card.className = 'cypher-query-card';
    card.innerHTML = `
      <div class="cypher-query-header">
        <h4 class="cypher-query-title">${escapeHtml(q.title)}</h4>
        <span class="cypher-op-pill">${escapeHtml(q.graphblas_operation)}</span>
      </div>
      <p class="cypher-query-desc">${escapeHtml(q.description)}</p>
      <pre class="cypher-code-block">${escapeHtml(q.cypher)}</pre>
    `;
    container.appendChild(card);
  });
}

async function runMCPTool(toolName) {
  const pid = selectedPatientId || 'PAT-102';
  const outPre = $('#mcp-json-output');
  $('#mcp-tool-executed-label').textContent = `Tool: ${toolName}`;
  outPre.textContent = 'Executing GraphBLAS traversal on FalkorDB...';

  const t0 = performance.now();
  let args = {};

  if (toolName === 'analyze_patient_vascular_tree') {
    args = { patient_id: pid };
  } else if (toolName === 'calculate_ischemic_blast_radius') {
    args = { artery_segment_id: 'SEG_LAD_PROX', plaque_id: 'PLQ-102-LAD-PROX' };
  } else if (toolName === 'find_vulnerable_plaques') {
    args = { min_vulnerability_score: 0.60 };
  } else if (toolName === 'simulate_plaque_rupture') {
    const plq = selectedPatientData?.plaques?.[0];
    args = { plaque_id: plq?.id || 'PLQ-102-LAD-PROX' };
  }

  try {
    const res = await fetch(`${API_BASE}/mcp/call`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: toolName, arguments: args })
    });
    const data = await res.json();
    const dt = (performance.now() - t0).toFixed(1);
    $('#mcp-latency').textContent = `${dt} ms (GraphBLAS)`;
    outPre.textContent = JSON.stringify(data.result, null, 2);
  } catch (err) {
    outPre.textContent = `Error executing tool: ${err.message}`;
  }
}

// ============================================================================
// CHATBOT & LANGGRAPH AGENT INTEGRATION
// ============================================================================

function setMode(mode) {
  currentMode = mode;
  $('#btn-mode-patient').classList.toggle('active', mode === 'patient');
  $('#btn-mode-doctor').classList.toggle('active', mode === 'doctor');

  // Update banner
  if (mode === 'doctor') {
    $('#mode-card-icon').textContent = '🩺';
    $('#mode-card-title').textContent = 'Interventional Cardiologist Cockpit Active';
    $('#mode-card-desc').textContent = 'Analyzing hemodynamic wall shear stress (WSS), carinal bifurcation angles, SYNTAX scoring, and catheter lab revascularization protocols.';
    $('#mode-card-pill').textContent = 'Clinical Specialist Mode';
  } else {
    $('#mode-card-icon').textContent = '💙';
    $('#mode-card-title').textContent = 'Patient Care Companion Active';
    $('#mode-card-desc').textContent = 'Providing empathetic, plain-English guidance on plaque vulnerability, medication regimens, heart-healthy nutrition, and daily safety precautions.';
    $('#mode-card-pill').textContent = 'Plain English Mode';
  }

  renderQuickPrompts();
}

function renderQuickPrompts() {
  elQuickQuestions.innerHTML = '';

  const prompts = (currentMode === 'doctor') ? [
    { label: '🔬 SYNTAX & Revascularization', msg: 'Provide SYNTAX score revascularization suitability, anatomy, and PCI recommendations.' },
    { label: '⚡ Carinal Shear Stress & WSS', msg: 'Analyze hemodynamic wall shear stress (WSS) and carinal bifurcation turbulence factor.' },
    { label: '🛡️ TCFA Cap Microns & OCT', msg: 'Evaluate thin-cap fibroatheroma (TCFA) cap thickness and IVUS/OCT imaging criteria.' },
    { label: '💊 Intensive Lipid Stabilization', msg: 'Outline PCSK9 inhibitor + high-intensity statin + ezetimibe plaque stabilization protocol.' },
    { label: '🚨 Code STEMI Cath Protocol', msg: 'Detail Code STEMI acute activation protocol and balloon thrombectomy workflow.' }
  ] : [
    { label: '🩺 Full Overview', msg: 'Give me my complete heart health overview' },
    { label: '💊 Treatment Plan', msg: 'What treatment and medicines do I need?' },
    { label: '🥗 Nutrition Guide', msg: 'What should I eat and what foods to avoid?' },
    { label: '🏃 Exercise Safety', msg: 'What exercise is safe for me right now?' },
    { label: '🚨 Warning Signs', msg: 'What are the emergency warning signs I should know?' },
    { label: '🩸 Blood Tests', msg: 'Explain my blood test results in simple words' }
  ];

  prompts.forEach(p => {
    const btn = document.createElement('button');
    btn.className = 'quick-btn';
    btn.textContent = p.label;
    btn.dataset.msg = p.msg;
    btn.addEventListener('click', () => {
      sendChatMessage(p.msg);
    });
    elQuickQuestions.appendChild(btn);
  });
}

function resetChat() {
  elChatMessages.innerHTML = `
    <div class="chat-msg bot-msg">
      <div class="msg-avatar bot-avatar">AI</div>
      <div class="msg-content">
        <p>I've loaded <strong>${selectedPatientData?.name || selectedPatientId}</strong>'s vascular profile. Analyzing their coronary data now...</p>
      </div>
    </div>
  `;
  elReasoningSection.style.display = 'none';
  elReasoningSteps.innerHTML = '';
}

function addUserMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'chat-msg user-msg';
  msg.innerHTML = `
    <div class="msg-avatar user-avatar">You</div>
    <div class="msg-content"><p>${escapeHtml(text)}</p></div>
  `;
  elChatMessages.appendChild(msg);
  scrollChatToBottom();
}

function addBotMessage(html) {
  const msg = document.createElement('div');
  msg.className = 'chat-msg bot-msg';
  msg.innerHTML = `
    <div class="msg-avatar bot-avatar">AI</div>
    <div class="msg-content">${renderMarkdown(html)}</div>
  `;
  elChatMessages.appendChild(msg);
  scrollChatToBottom();
}

function addTypingIndicator() {
  const msg = document.createElement('div');
  msg.className = 'chat-msg bot-msg';
  msg.id = 'typing-indicator';
  msg.innerHTML = `
    <div class="msg-avatar bot-avatar">AI</div>
    <div class="msg-content">
      <div class="typing-indicator">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  elChatMessages.appendChild(msg);
  scrollChatToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

function scrollChatToBottom() {
  setTimeout(() => {
    elChatMessages.scrollTop = elChatMessages.scrollHeight;
  }, 50);
}

async function sendChatMessage(message) {
  if (!selectedPatientId || !message.trim() || isChatLoading) return;

  isChatLoading = true;
  elChatSendBtn.disabled = true;
  addUserMessage(message);
  addTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_id: selectedPatientId,
        message: message,
        mode: currentMode
      })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error ${res.status}`);
    }

    const result = await res.json();
    removeTypingIndicator();

    if (result.reasoning_steps && result.reasoning_steps.length > 0) {
      renderReasoningSteps(result.reasoning_steps);
    }

    if (result.final_response) {
      addBotMessage(result.final_response);
    }

  } catch (err) {
    removeTypingIndicator();
    addBotMessage(`<p style="color:var(--accent-rose);">⚠️ Error: ${escapeHtml(err.message)}</p>`);
  } finally {
    isChatLoading = false;
    elChatSendBtn.disabled = false;
  }
}

function renderReasoningSteps(steps) {
  elReasoningSection.style.display = 'block';
  elReasoningSteps.innerHTML = '';

  steps.forEach((step, i) => {
    const el = document.createElement('div');
    el.className = 'reasoning-step';
    el.style.animationDelay = `${i * 0.15}s`;
    el.innerHTML = `
      <div class="step-number">${i + 1}</div>
      <div class="step-text">${escapeHtml(step)}</div>
    `;
    elReasoningSteps.appendChild(el);
  });
}

// ============================================================================
// TAB NAVIGATION & SPLIT VIEW
// ============================================================================

function switchTab(tabId) {
  activeTab = tabId;
  $$('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });

  $$('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === tabId);
  });

  if (tabId === 'tab-cypher') {
    runMCPTool('analyze_patient_vascular_tree');
  }
}

function toggleSplitView() {
  isSplitView = !isSplitView;
  $('#main-content').classList.toggle('split-active', isSplitView);
  $('#btn-split-toggle').classList.toggle('active', isSplitView);
}

// ============================================================================
// UPLOAD PATIENT & FALKORDB INSPECTOR
// ============================================================================

function openUploadModal() {
  elUploadModal.style.display = 'flex';
}

function closeUploadModal() {
  elUploadModal.style.display = 'none';
}

function openInspectorModal() {
  const pid = selectedPatientId || 'PAT-102';
  fetch(`${API_BASE}/api/cypher-queries/${pid}`)
    .then(r => r.json())
    .then(data => {
      const body = $('#inspector-modal-body');
      let html = `<div style="margin-bottom:16px;">
        <p style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:12px;">Active Graph: <strong>${data.graph_name}</strong> · Mode: <strong>${data.engine_mode}</strong></p>
      </div>`;

      (data.queries || []).forEach(q => {
        html += `
          <div class="cypher-query-card" style="margin-bottom:14px;">
            <div class="cypher-query-header">
              <h4 class="cypher-query-title">${escapeHtml(q.title)}</h4>
              <span class="cypher-op-pill">${escapeHtml(q.graphblas_operation)}</span>
            </div>
            <p class="cypher-query-desc">${escapeHtml(q.description)}</p>
            <pre class="cypher-code-block">${escapeHtml(q.cypher)}</pre>
          </div>
        `;
      });

      body.innerHTML = html;
      elInspectorModal.style.display = 'flex';
    });
}

function closeInspectorModal() {
  elInspectorModal.style.display = 'none';
}

async function handleUploadSubmit(e) {
  e.preventDefault();

  const payload = {
    name: $('#inp-name').value,
    age: parseInt($('#inp-age').value) || 52,
    gender: $('#inp-gender').value,
    smoker: $('#inp-smoker').value === 'true',
    diabetes: $('#inp-diabetes').value === 'true',
    hypertension: $('#inp-hypertension').value === 'true',
    clinical_presentation: 'Uploaded patient CCTA and lab profile',
    biomarkers: {
      apob_mg_dl: parseFloat($('#inp-apob').value) || 120.0,
      hscrp_mg_l: parseFloat($('#inp-hscrp').value) || 2.5,
      lpa_nmol_l: parseFloat($('#inp-lpa').value) || 60.0,
      ldl_c_mg_dl: parseFloat($('#inp-ldl').value) || 110.0,
      troponin_i_ng_ml: 0.015
    },
    plaques: [{
      id: `PLQ-UPLOAD-${Date.now()}`,
      artery_segment_id: $('#inp-artery').value,
      stenosis_pct: parseFloat($('#inp-stenosis').value) || 38,
      plaque_type: $('#inp-plaque-type').value,
      cap_thickness_um: parseFloat($('#inp-cap').value) || 45,
      core_lipid_ratio: 0.65,
      calcification_score: 0.8,
      vulnerability_score: 0.95,
      rupture_risk_category: 'CRITICAL'
    }]
  };

  try {
    const res = await fetch(`${API_BASE}/api/upload-patient-json`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error('Upload failed');
    const result = await res.json();
    const newPid = result.patient_id;

    payload.id = newPid;
    allPatients.unshift(payload);
    renderPatientList();
    closeUploadModal();
    await selectPatient(newPid);

  } catch (err) {
    alert('Upload error: ' + err.message);
  }
}

// ============================================================================
// EVENT LISTENERS SETUP
// ============================================================================

function setupEventListeners() {
  // Mode toggles
  $('#btn-mode-patient').addEventListener('click', () => setMode('patient'));
  $('#btn-mode-doctor').addEventListener('click', () => setMode('doctor'));

  // Top action buttons
  $('#btn-top-simulate').addEventListener('click', simulatePlaqueRupture);
  $('#btn-top-inspector').addEventListener('click', openInspectorModal);
  $('#btn-sim-rupture-canvas').addEventListener('click', simulatePlaqueRupture);

  // STEMI HUD buttons
  $('#btn-deploy-stent').addEventListener('click', deployStentAngioplasty);
  $('#btn-reset-sim').addEventListener('click', resetSimulationState);

  // Inspector card close
  $('#insp-close').addEventListener('click', () => {
    elVesselInspectorCard.style.display = 'none';
  });

  // Tab navigation
  $$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      switchTab(btn.dataset.tab);
    });
  });

  // Split view toggle
  $('#btn-split-toggle').addEventListener('click', toggleSplitView);

  // Filter pills
  $$('.filter-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      $$('.filter-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      activeTreeFilter = pill.dataset.filter;
      renderCoronaryTree();
    });
  });

  // MCP tool runner buttons
  $$('.btn-mcp-run').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.btn-mcp-run').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      runMCPTool(btn.dataset.tool);
    });
  });

  // Chat send
  elChatSendBtn.addEventListener('click', () => {
    const msg = elChatInput.value.trim();
    if (msg) {
      elChatInput.value = '';
      sendChatMessage(msg);
    }
  });

  elChatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const msg = elChatInput.value.trim();
      if (msg) {
        elChatInput.value = '';
        sendChatMessage(msg);
      }
    }
  });

  // Modals
  $('#btn-upload-toggle').addEventListener('click', openUploadModal);
  $('#modal-close').addEventListener('click', closeUploadModal);
  elUploadModal.addEventListener('click', (e) => {
    if (e.target === elUploadModal) closeUploadModal();
  });

  $('#btn-sidebar-inspector').addEventListener('click', openInspectorModal);
  $('#inspector-modal-close').addEventListener('click', closeInspectorModal);
  elInspectorModal.addEventListener('click', (e) => {
    if (e.target === elInspectorModal) closeInspectorModal();
  });

  $('#upload-form').addEventListener('submit', handleUploadSubmit);

  // Guide modal listeners
  $('#btn-open-guide-modal')?.addEventListener('click', openGuideModal);
  $('#guide-modal-close')?.addEventListener('click', closeGuideModal);
  elGuideModal?.addEventListener('click', (e) => {
    if (e.target === elGuideModal) closeGuideModal();
  });

  // Vessel zoom modal listeners
  $('#zoom-modal-close')?.addEventListener('click', closeVesselZoomModal);
  elVesselZoomModal?.addEventListener('click', (e) => {
    if (e.target === elVesselZoomModal) closeVesselZoomModal();
  });

  $('#btn-zoom-closer')?.addEventListener('click', () => {
    zoomCanvas(0.55);
  });

  $('#btn-zoom-reset-from-modal')?.addEventListener('click', () => {
    resetCanvasZoom();
    closeVesselZoomModal();
  });

  $('#btn-zoom-trigger-rupture')?.addEventListener('click', () => {
    closeVesselZoomModal();
    simulatePlaqueRupture();
  });

  $('#btn-zoom-ask-ai')?.addEventListener('click', () => {
    closeVesselZoomModal();
    switchTab('tab-chat');
    const segName = selectedSegmentId ? (ARTERY_PATHS[selectedSegmentId]?.name || selectedSegmentId) : 'coronary artery';
    sendChatMessage(`What is the clinical rupture danger and treatment plan for my ${segName}?`);
  });

  // Canvas Zoom Buttons
  $('#btn-zoom-in')?.addEventListener('click', () => zoomCanvas(0.7));
  $('#btn-zoom-out')?.addEventListener('click', () => zoomCanvas(1.4));
  $('#btn-zoom-reset')?.addEventListener('click', () => resetCanvasZoom());
  $('#btn-zoom-focus')?.addEventListener('click', () => focusOnPlaque());

  // Canvas Mouse Wheel Zoom & Drag Pan
  if (elCoronarySvg) {
    elCoronarySvg.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = elCoronarySvg.getBoundingClientRect();
      const mouseSvgX = currentViewBox.x + ((e.clientX - rect.left) / rect.width) * currentViewBox.w;
      const mouseSvgY = currentViewBox.y + ((e.clientY - rect.top) / rect.height) * currentViewBox.h;
      const factor = e.deltaY < 0 ? 0.75 : 1.3;
      zoomCanvas(factor, { x: mouseSvgX, y: mouseSvgY });
    }, { passive: false });

    elCoronarySvg.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      isPanning = true;
      panStart = { x: e.clientX, y: e.clientY };
      panViewBoxStart = { ...currentViewBox };
      elCoronarySvg.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
      if (!isPanning || !elCoronarySvg) return;
      const dx = (e.clientX - panStart.x);
      const dy = (e.clientY - panStart.y);
      const rect = elCoronarySvg.getBoundingClientRect();
      const scaleX = currentViewBox.w / rect.width;
      const scaleY = currentViewBox.h / rect.height;

      const newX = Math.max(0, Math.min(BASE_VIEWBOX.w - currentViewBox.w, panViewBoxStart.x - dx * scaleX));
      const newY = Math.max(0, Math.min(BASE_VIEWBOX.h - currentViewBox.h, panViewBoxStart.y - dy * scaleY));

      currentViewBox.x = newX;
      currentViewBox.y = newY;
      updateViewBoxDisplay();
    });

    window.addEventListener('mouseup', () => {
      if (isPanning) {
        isPanning = false;
        if (elCoronarySvg) elCoronarySvg.style.cursor = 'default';
      }
    });
  }
}


// ============================================================================
// UTILITIES
// ============================================================================

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML;
}

function renderMarkdown(text) {
  let html = text || '';

  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^---$/gm, '<hr>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

  // Markdown tables
  const tableRegex = /(\|.+\|\n)+/g;
  html = html.replace(tableRegex, (tableBlock) => {
    const rows = tableBlock.trim().split('\n').filter(r => r.trim());
    if (rows.length < 2) return tableBlock;

    let tableHtml = '<table>';
    rows.forEach((row, idx) => {
      if (/^\|[\s\-|:]+\|$/.test(row)) return;
      const cells = row.split('|').filter(c => c.trim() !== '');
      const tag = idx === 0 ? 'th' : 'td';
      if (idx === 0) tableHtml += '<thead><tr>';
      else if (idx === 1) tableHtml += '<tbody><tr>';
      else tableHtml += '<tr>';

      cells.forEach(cell => {
        tableHtml += `<${tag}>${cell.trim()}</${tag}>`;
      });
      tableHtml += idx === 0 ? '</tr></thead>' : '</tr>';
    });
    tableHtml += '</tbody></table>';
    return tableHtml;
  });

  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.+<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`);

  html = html.replace(/\n\n/g, '</p><p>');
  if (!html.startsWith('<')) {
    html = `<p>${html}</p>`;
  }

  return html;
}

document.addEventListener('DOMContentLoaded', init);
