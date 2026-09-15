import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { checkQuality, predict, generateReport } from '../api/client';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const LABELS = ["No DR","Mild DR","Moderate DR","Severe DR","Proliferative DR"];
const TIERS = ["MONITOR","MONITOR","ENGAGE","ACT NOW","ACT NOW"];
const TIER_COLORS = ["#10b981","#10b981","#f59e0b","#d94f5c","#d94f5c"];
const BAR_COLORS = ["#10b981","#06b6d4","#f59e0b","#f97316","#d94f5c"];


function generateResultHtml(data) {
  const i = data.predicted_class;
  const color = TIER_COLORS[i];
  return `
    ${data.confidence_flag === 'low' ? `
    <div style="background:#451a03;border:1px solid #f59e0b66;border-radius:8px;padding:10px 14px;margin-bottom:12px;display:flex;align-items:center;gap:8px;">
      <span style="font-size:1.1rem;line-height:1;">⚠️</span>
      <span style="color:#fbbf24;font-size:0.82rem;font-weight:600;">Low confidence — please have this case reviewed by an ophthalmologist</span>
    </div>` : ''}
    <div class="risk-pill" style="background:${color}18;border:1px solid ${color}44">
      <div class="risk-dot" style="background:${color}"></div>
      <span style="color:${color}">${TIERS[i]}</span>
    </div>
    <div class="result-diagnosis" style="color:${color}">${LABELS[i]}</div>
    <div class="result-conf-row">
      <span class="result-conf-text">${(data.confidence*100).toFixed(1)}% confidence</span>
      <div class="conf-track">
        <div class="conf-fill" id="confFill" style="background:${color};width:${(data.confidence*100).toFixed(1)}%"></div>
      </div>
    </div>
    <div style="font-size:0.72rem; color:var(--muted); margin-bottom:10px;">Input image quality score: <b style="color:var(--text)">${data.quality_score ? (data.quality_score*100).toFixed(0) : 'N/A'}/100</b></div>
    <div class="result-desc">${data.description}</div>
    <div class="divider"></div>
    <div class="scores-title">Confidence Scores</div>
    ${data.probabilities.map((p,j) => `
      <div class="bar-row">
        <div class="bar-label">
          <span>${LABELS[j]}</span>
          <span>${(p*100).toFixed(1)}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" id="bar${j}" style="background:${BAR_COLORS[j]};width:${(p*100).toFixed(1)}%"></div>
        </div>
      </div>`).join('')}
  `;
}

export default function Home() {
  const navigate = useNavigate();
  const [viewState, setViewState] = useState('intro'); // intro, upload, qualityReject, results
  const [file, setFile] = useState(null);
  const [scanPreview, setScanPreview] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeText, setAnalyzeText] = useState('Select an image to begin');
  
  const [qData, setQData] = useState(null);
  const [reportData, setReportData] = useState(null);
  
  const hnavRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    document.querySelectorAll('.lp-reveal').forEach(el => {
      gsap.to(el, {
        opacity: 1, y: 0, duration: 0.6, ease: 'power2.out',
        scrollTrigger: { trigger: el, start: 'top 90%' }
      });
    });
    document.querySelectorAll('.lp-stat .num').forEach(el => {
      const target = parseFloat(el.dataset.count);
      const suffix = el.dataset.suffix || '';
      ScrollTrigger.create({
        trigger: el, start: 'top 92%', once: true,
        onEnter: () => gsap.to({ v: 0 }, {
          v: target, duration: 1.1, ease: 'power2.out',
          onUpdate: function () { el.textContent = this.targets()[0].v.toFixed(target % 1 === 0 ? 0 : 1) + suffix; }
        })
      });
    });
  }, [viewState]);

  const navTo = (target) => {
    if (hnavRef.current) hnavRef.current.classList.remove('active');
    setViewState('intro');
    setTimeout(() => {
      if (target === 'home') window.scrollTo({top:0, behavior:'smooth'});
      else document.getElementById(target)?.scrollIntoView({behavior:'smooth'});
    }, 100);
  };

  const startApp = () => {
    setViewState('upload');
    window.scrollTo({top:0, behavior:'smooth'});
  };

  const goHome = () => {
    setViewState('intro');
    window.scrollTo({top:0, behavior:'smooth'});
  };

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    const reader = new FileReader();
    reader.onload = ev => setScanPreview(ev.target.result);
    reader.readAsDataURL(f);
    setAnalyzeText('Analyze Image');
  };

  const doAnalyze = async () => {
    if (!file) return;
    setAnalyzing(true);
    setAnalyzeText('Checking image quality...');
    
    try {
      const qres = await checkQuality(file);
      if (!qres.is_usable) {
        setQData(qres);
        setViewState('qualityReject');
        setAnalyzing(false);
        setAnalyzeText('Analyze Image');
        return;
      }
      
      setAnalyzeText('Analyzing...');
      const data = await predict(file);
      data.quality_score = qres.quality_score;
      setReportData(data);
      setViewState('results');
    } catch (e) {
      alert('Error analyzing image. Please try again.');
    }
    setAnalyzing(false);
    setAnalyzeText('Analyze Image');
  };

  const doDownloadReport = async () => {
    if (!reportData) return;
    try {
      const blob = await generateReport(reportData);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'DR_Screening_Report.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      alert('Could not generate report. Please try again.');
    }
  };

  return (
    <div className="home-container">
      
<header className="header">
<div className="logo">
<svg fill="none" height="20" viewBox="0 0 24 24" width="20" xmlns="http://www.w3.org/2000/svg">
<path d="M1 12C1 12 5 4 12 4C19 4 23 12 23 12C23 12 19 20 12 20C5 20 1 12 1 12Z" stroke="white" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
<circle cx="12" cy="12" r="3.5" stroke="white" strokeWidth="2" />
</svg>
</div>
<div>
<div className="text-[1.05rem] font-extrabold tracking-[-0.02em]">Netra</div>
<div className="hsub">AI-Assisted Diabetic Retinopathy Screening</div>
</div>
<nav className="hnav" ref={hnavRef}>
<a onClick={() => navTo('home')}>Home</a>
<a onClick={() => navigate("/patients")} style={{cursor:"pointer"}}>Patients</a>
<a onClick={() => navTo('howItWorks')}>How It Works</a>
<a onClick={() => navTo('capabilities')}>Capabilities</a>
</nav>
<button className="hnav-cta" id="hnavCta" onClick={() => startApp()}>Start Screening</button>
<div className="hamburger" onClick={() => { if(hnavRef.current) hnavRef.current.classList.toggle('active') }}>
<svg fill="none" height="24" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" width="24">
<path d="M4 6h16M4 12h16M4 18h16" />
</svg>
</div>
</header>
<main className="max-w-[1560px] mx-auto pt-8 px-14 pb-[60px]">
{/*  INTRO / LANDING STATE  */}
<div id="introState" style={{ display: viewState === 'intro' ? 'block' : 'none' }}>
<section className="lp-hero">
<div className="lp-float lp-f1 lp-reveal">🛡️</div>
<div className="lp-float lp-f2 lp-reveal">🔍</div>
<div className="lp-float lp-f3 lp-reveal">📊</div>
<div className="lp-badge lp-reveal"><span className="dot"></span>AI-ASSISTED RETINAL SCREENING · EFFICIENTNETB3</div>
<h1 className="lp-reveal">
<span className="lp-serif line1">See Diabetic Retinopathy</span>
<span className="lp-serif line2">Before It Steals Sight.</span>
</h1>
<p className="lp-sub lp-reveal">Netra reads retinal fundus photographs the way a specialist would — grading severity, checking image quality first, and showing exactly what it saw so nothing is missed and nothing is hidden.</p>
<div className="lp-ctas lp-reveal">
<button className="w-full mt-[14px] bg-gradient-to-br from-[#0e9f92] to-[#087267] border-none rounded-[11px] text-white text-[0.88rem] font-bold p-[14px] cursor-pointer transition-all duration-200 flex items-center justify-center gap-2 shadow-[0_4px_18px_#0e9f9228] hover:-translate-y-[1px] hover:shadow-[0_6px_22px_#0e9f9240] disabled:opacity-45 disabled:cursor-not-allowed disabled:shadow-none disabled:transform-none" onClick={() => startApp()} style={{'width': 'auto', 'marginTop': '0', 'padding': '13px 22px'}}>Start Screening →</button>
<button className="lp-btn-secondary" onClick={() => navTo('howNetraWorks')}>See How It Works</button>
</div>
</section>
<div className="lp-stats">
<div className="lp-stat lp-reveal"><div className="num" data-count="96.7" data-suffix="%">0</div><div className="lbl">Sensitivity</div></div>
<div className="lp-stat lp-reveal"><div className="num" data-count="3">0</div><div className="lbl">Screening Stages</div></div>
<div className="lp-stat lp-reveal"><div className="num" data-count="94">0</div><div className="lbl">Validation Images</div></div>
</div>
<div className="lp-section">
<div className="lp-label lp-reveal">THE SCREENING BLINDSPOT</div>
<div className="lp-title lp-reveal">Manual Screening Doesn't Scale. Early DR Often Goes Unnoticed.</div>
<div className="lp-problem-grid">
<div className="lp-problem-card lp-reveal">
<div className="lp-problem-icon">
<svg fill="none" height="28" stroke="var(--text)" strokeWidth="1.8" viewBox="0 0 24 24" width="28">
<path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" strokeLinecap="round" strokeLinejoin="round" />
</svg>
</div>
<h3>Manual Grading</h3>
<p>Relies on specialist availability — results can take days and vary between readers.</p>
</div>
<div className="lp-problem-card mid lp-reveal">
<div className="lp-problem-icon">
<svg fill="none" height="28" stroke="var(--text)" strokeWidth="1.8" viewBox="0 0 24 24" width="28">
<path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" strokeLinecap="round" strokeLinejoin="round" />
<path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" strokeLinecap="round" strokeLinejoin="round" />
</svg>
</div>
<h3>Silent Progression</h3>
<p>Diabetic retinopathy often shows no symptoms until vision loss has already begun.</p>
</div>
<div className="lp-problem-card hi lp-reveal">
<span className="lp-tag">The Critical Question</span>
<h3>Does this retina show signs of DR right now?</h3>
<p>Netra answers with a graded severity, visual evidence, and a clear referral recommendation.</p>
</div>
</div>
</div>
<div className="lp-section" id="howNetraWorks">
<div className="lp-label lp-reveal">SCREENING PIPELINE</div>
<div className="lp-title lp-reveal">How Netra Works</div>
<div className="lp-desc lp-reveal">Four stages turn a single retinal photo into a clinically useful screening decision.</div>
<div className="lp-pipe-grid">
<div className="lp-pipe-card lp-reveal">
<div className="lp-visual lp-v-blur"><div className="lp-v-retina"></div><div className="lp-v-badge">BLURRY</div></div>
<div className="lp-pipe-top"><span className="lp-pipe-num">01</span><span className="lp-pipe-badge">ACTIVE</span></div>
<h4>Quality Check</h4>
<p>Automatically rejects blurry or poorly-lit scans before they reach the model.</p>
</div>
<div className="lp-pipe-card lp-reveal">
<div className="lp-visual"><div className="lp-v-heat"></div></div>
<div className="lp-pipe-top"><span className="lp-pipe-num">02</span><span className="lp-pipe-badge">ACTIVE</span></div>
<h4>AI Grading</h4>
<p>EfficientNetB3 grades severity across five ICDR classes, with calibrated confidence.</p>
</div>
<div className="lp-pipe-card lp-reveal">
<div className="lp-visual">
<div className="lp-v-retina"></div>
<div className="lp-v-lesion">
<div className="dot" style={{'top': '27px', 'left': '58px', 'background': '#d94f5c'}}></div>
<div className="dot" style={{'top': '51px', 'left': '72px', 'background': '#d94f5c'}}></div>
<div className="dot" style={{'top': '65px', 'left': '41px', 'background': '#d9a441'}}></div>
</div>
</div>
<div className="lp-pipe-top"><span className="lp-pipe-num">03</span><span className="lp-pipe-badge">ACTIVE</span></div>
<h4>Lesion Detection</h4>
<p>Highlights lesion candidates and a Grad-CAM heatmap for visual interpretability.</p>
</div>
<div className="lp-pipe-card lp-reveal">
<div className="lp-visual lp-v-report"><div className="doc"><div className="badge-ok">✓</div></div></div>
<div className="lp-pipe-top"><span className="lp-pipe-num">04</span><span className="lp-pipe-badge">ACTIVE</span></div>
<h4>Clinical Report</h4>
<p>Packages the result into a downloadable PDF with a referral recommendation.</p>
</div>
</div>
</div>
<div className="lp-section" id="capabilities">
<div className="lp-label lp-reveal">SCREENING ENGINE</div>
<div className="lp-title lp-reveal">Key Capabilities</div>
<div className="lp-cap-grid">
<div className="lp-cap-item lp-reveal"><div className="lp-cap-icon">✓</div><h4>Automated Quality Gating</h4><p>Rejects unusable images before grading, so results are never built on bad data.</p></div>
<div className="lp-cap-item lp-reveal"><div className="lp-cap-icon">◎</div><h4>Explainable Grad-CAM Heatmaps</h4><p>Visualizes exactly which retinal regions drove the model's decision.</p></div>
<div className="lp-cap-item lp-reveal"><div className="lp-cap-icon">◆</div><h4>Lesion Candidate Overlay</h4><p>Classical CV highlights dark and bright lesion candidates for visual review.</p></div>
<div className="lp-cap-item lp-reveal"><div className="lp-cap-icon">⚑</div><h4>Calibrated Confidence Flagging</h4><p>Low-confidence predictions are automatically routed for human review.</p></div>
<div className="lp-cap-item lp-reveal"><div className="lp-cap-icon">▤</div><h4>Honest Validation Reporting</h4><p>Real external validation metrics, including known limitations, shown transparently.</p></div>
<div className="lp-cap-item lp-reveal"><div className="lp-cap-icon">⇩</div><h4>One-Click Clinical Report</h4><p>Generates a shareable PDF with grade, evidence, and referral guidance.</p></div>
</div>
</div>
<div className="lp-cta-band lp-reveal">
<div>
<div className="lp-cta-label">Get Started</div>
<h3 className="lp-serif">From a Photo to a Screening Decision.</h3>
<p>Upload a retinal fundus image and get a graded, explainable result in seconds.</p>
</div>
<button className="lp-btn-launch" onClick={() => startApp()}>Launch Screening Tool <span className="arrow">→</span></button>
</div>
<div className="lp-footnote">NETRA · AI-ASSISTED DECISION SUPPORT · NOT A DIAGNOSIS</div>
</div>
{/*  UPLOAD STATE  */}
<div id="uploadState" style={{ display: viewState === 'upload' ? 'block' : 'none' }}>
<div className="bg-white border border-[#d6e0e4] rounded-[16px] p-[22px] shadow-[0_8px_22px_rgba(20,43,58,0.05)]" style={{'marginBottom': '18px'}}>
<div className="text-[0.67rem] font-bold text-[#8896a1] uppercase tracking-[0.1em] mb-[14px]">Upload Retinal Image</div>
<div className="border-2 border-dashed border-[#d6e0e4] rounded-[14px] pt-[52px] px-6 pb-[52px] text-center cursor-pointer transition-all duration-200 relative bg-[#f2f6f5] hover:border-[#0e9f92] hover:bg-[#0e9f920a]" id="uploadZone" onClick={() => fileInputRef.current?.click()}>
<input accept="image/*" id="imageInput" type="file"/>
<div className="text-[1rem] font-bold text-[#172033] mb-[6px]">Upload a retinal fundus image</div>
<div className="text-[0.8rem] text-[#8896a1] leading-[1.6]"><span>Click to browse</span> or drag and drop</div>
<div className="text-[0.72rem] text-[#8896a1] mt-[8px] opacity-70">PNG, JPG, JPEG supported</div>
{file && <div className="upload-filename" id="uploadFileName" style={{display: 'inline-block'}}>✓ {file.name}</div>}
</div>
<button className="w-full mt-[14px] bg-gradient-to-br from-[#0e9f92] to-[#087267] border-none rounded-[11px] text-white text-[0.88rem] font-bold p-[14px] cursor-pointer transition-all duration-200 flex items-center justify-center gap-2 shadow-[0_4px_18px_#0e9f9228] hover:-translate-y-[1px] hover:shadow-[0_6px_22px_#0e9f9240] disabled:opacity-45 disabled:cursor-not-allowed disabled:shadow-none disabled:transform-none" disabled="" id="analyzeBtn" onClick={(e) => { e.stopPropagation(); doAnalyze(); }} disabled={!file || analyzing}>
<span id="btnText">{analyzeText}</span>
{analyzing && <div className="spinner" id="spinner"></div>}
</button>
</div>
</div>
{/*  QUALITY REJECT STATE  */}
<div className="bg-white border border-[#d6e0e4] rounded-[16px] p-[22px] shadow-[0_8px_22px_rgba(20,43,58,0.05)]" id="qualityRejectState" style={{ display: viewState === 'qualityReject' ? 'block' : 'none', borderColor: '#d94f5c' }}>
<div className="text-[0.67rem] font-bold text-[#8896a1] uppercase tracking-[0.1em] mb-[14px]" style={{'color': '#d94f5c'}}>Image Rejected — Quality Check Failed</div>
<div style={{'fontSize': '0.85rem', 'color': 'var(--text)', 'marginBottom': '10px'}}>
      Quality score: <span id="rejectScore" style={{fontWeight: '700'}}>{qData ? (qData.quality_score * 100).toFixed(0) + '/100' : ''}</span>
</div>
<div style={{'fontSize': '0.82rem', 'color': '#fca5a5', 'marginBottom': '10px'}}>
      Reason: <span id="rejectReason">{qData?.reason || 'Image quality insufficient for grading.'}</span>
</div>
<div className="result-desc" style={{'borderLeftColor': '#d94f5c'}}>
<span id="rejectGuidance">{qData?.clinical_guidance || 'Please recapture the image.'}</span>
</div>
<button className="w-full mt-[14px] bg-gradient-to-br from-[#0e9f92] to-[#087267] border-none rounded-[11px] text-white text-[0.88rem] font-bold p-[14px] cursor-pointer transition-all duration-200 flex items-center justify-center gap-2 shadow-[0_4px_18px_#0e9f9228] hover:-translate-y-[1px] hover:shadow-[0_6px_22px_#0e9f9240] disabled:opacity-45 disabled:cursor-not-allowed disabled:shadow-none disabled:transform-none" onClick={() => { setFile(null); setViewState('upload'); }} style={{'marginTop': '14px'}}>Recapture Image</button>
</div>
{/*  RESULTS STATE  */}
<div id="resultsState" style={{ display: viewState === 'results' ? 'block' : 'none' }}>
{/*  Severity Scale  */}
<div className="flex mb-[18px] rounded-[11px] overflow-hidden border border-[#d6e0e4]" id="severityScale">
<div className="sev-item" id="sev0" style={reportData?.predicted_class === 0 ? {background: "#052e16", color: "#10b981", borderBottom: "2px solid #10b981"} : {}}><div className="sev-dot" style={{opacity: 1}}></div>No DR</div>
<div className="sev-item" id="sev1" style={reportData?.predicted_class === 1 ? {background: "#052e16", color: "#10b981", borderBottom: "2px solid #10b981"} : {}}><div className="sev-dot" style={{opacity: 1}}></div>Mild</div>
<div className="sev-item" id="sev2" style={reportData?.predicted_class === 2 ? {background: "#1c1408", color: "#f59e0b", borderBottom: "2px solid #f59e0b"} : {}}><div className="sev-dot" style={{opacity: 1}}></div>Moderate</div>
<div className="sev-item" id="sev3" style={reportData?.predicted_class === 3 ? {background: "#1c0a0a", color: "#d94f5c", borderBottom: "2px solid #d94f5c"} : {}}><div className="sev-dot" style={{opacity: 1}}></div>Severe</div>
<div className="sev-item" id="sev4" style={reportData?.predicted_class === 4 ? {background: "#1c0a0a", color: "#d94f5c", borderBottom: "2px solid #d94f5c"} : {}}><div className="sev-dot" style={{opacity: 1}}></div>Proliferative</div>
</div>
{/*  Top Grid: Scan + Result  */}
<div className="top-grid">
<div className="bg-white border border-[#d6e0e4] rounded-[16px] p-[22px] shadow-[0_8px_22px_rgba(20,43,58,0.05)]">
<div className="text-[0.67rem] font-bold text-[#8896a1] uppercase tracking-[0.1em] mb-[14px]">Retinal Scan</div>
<div className="scan-img-wrap">
<img alt="Retinal scan" id="scanPreview"/>
</div>
<button className="w-full mt-[10px] bg-transparent border border-[#d6e0e4] rounded-[10px] text-[#8896a1] text-[0.78rem] font-medium p-[10px] cursor-pointer transition-all duration-200 hover:border-[#0e9f92] hover:text-[#087267]" onClick={() => { setFile(null); setViewState('upload'); }}>Upload another image</button>
</div>
<div className="bg-white border border-[#d6e0e4] rounded-[16px] p-[22px] shadow-[0_8px_22px_rgba(20,43,58,0.05)]" id="resultCard">
<div className="text-[0.67rem] font-bold text-[#8896a1] uppercase tracking-[0.1em] mb-[14px]">Classification Result</div>
<div id="resultBody" dangerouslySetInnerHTML={{ __html: reportData ? generateResultHtml(reportData) : '' }}></div>
</div>
</div>
{/*  Grad-CAM  */}
<div className="gradcam-card bg-white border border-[#d6e0e4] rounded-[16px] p-[22px] shadow-[0_8px_22px_rgba(20,43,58,0.05)]" id="gradcamCard" style={{ display: reportData ? 'block' : 'none' }}>
<div className="text-[0.67rem] font-bold text-[#8896a1] uppercase tracking-[0.1em] mb-[14px]">
        Model Attention — Grad-CAM Visualization
        <span id="gradcamTargetLabel" style={{float: 'right', color: 'var(--accent2)', textTransform: 'none', fontWeight: 'bold'}}>{reportData ? 'Target: ' + reportData.gradcam_target_class : ''}</span>
</div>
<div className="gradcam-grid">
<div>
<div className="rounded-[10px] overflow-hidden border border-[#d6e0e4] w-full bg-black flex items-center justify-center"><img alt="Original scan" id="origImg"/></div>
<div className="gcam-label">Original Retinal Scan</div>
</div>
<div>
<div className="rounded-[10px] overflow-hidden border border-[#d6e0e4] w-full bg-black flex items-center justify-center">
<img alt="Attention heatmap" id="camImg"/>
<div id="gradcamUnavailableMsg" style={{'display': 'none', 'height': '180px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'background': 'var(--surface2)', 'color': 'var(--muted)', 'fontStyle': 'italic'}}>Grad-CAM unavailable</div>
</div>
<div className="gcam-label">
            AI Attention Heatmap — <span style={{'background': 'linear-gradient(to right, blue, cyan, green, yellow, red)', 'WebkitBackgroundClip': 'text', 'color': 'transparent', 'fontWeight': 'bold'}}>Low to High Attention</span>
</div>
</div>
<div>
<div className="rounded-[10px] overflow-hidden border border-[#d6e0e4] w-full bg-black flex items-center justify-center"><img alt="Lesion overlay" id="lesionImg"/></div>
<div className="gcam-label" id="lesionLabel">{reportData?.lesion_count === -1 ? "Lesion Overlay — Lesion detection unavailable" : reportData?.lesion_count === 0 ? "Lesion Overlay — No candidate lesions detected by this detector; this does not rule out DR." : "Lesion Overlay — " + reportData?.lesion_count + " candidate regions flagged (red=dark, yellow=bright)"}</div>
</div>
</div>
<div className="mt-[14px] py-[11px] px-[14px] bg-[#eef7f6] border border-[#08726733] rounded-[10px] text-[0.76rem] text-[#087267] leading-[1.65]">
    Red zones indicate regions receiving higher attention from the model during classification. 
    This visualization provides an additional interpretability signal alongside the model prediction.
</div>
</div>
{/*  Model Performance  */}
<div className="perf-card bg-white border border-[#d6e0e4] rounded-[16px] p-[22px] shadow-[0_8px_22px_rgba(20,43,58,0.05)]">
<div className="text-[0.67rem] font-bold text-[#8896a1] uppercase tracking-[0.1em] mb-[14px]">Model Performance — External Validation (IDRiD, IQA-Accepted)</div>
<div className="perf-grid">
<div className="perf-metric"><div className="perf-val">96.7%</div><div className="perf-lbl">Sensitivity (Referable DR)</div></div>
<div className="perf-metric"><div className="perf-val">0.852</div><div className="perf-lbl">QWK (Cohen's Kappa)</div></div>
<div className="perf-metric"><div className="perf-val">94</div><div className="perf-lbl">External Validation Images</div></div>
</div>
<div style={{'background': '#0a1929', 'border': '1px solid #3b82f644', 'borderRadius': '8px', 'padding': '10px 14px', 'marginTop': '12px', 'fontSize': '0.78rem', 'color': '#93c5fd', 'lineHeight': '1.4'}}>
<b>Design philosophy — screening-first:</b> Like mammography and other high-stakes screening tools, this model is tuned to prioritize catching every possible case of DR (96.7% sensitivity) rather than minimizing false alarms. Flagged cases are designed to route to a human ophthalmologist for confirmation — never as a standalone diagnosis. Threshold tuning to improve specificity is an active, ongoing area of the project (see validation_sim/README.md for full methodology).
      </div>
</div>
{/*  AI Clinical Intelligence  */}
<div className="ai-card bg-white border border-[#d6e0e4] rounded-[16px] p-[22px] shadow-[0_8px_22px_rgba(20,43,58,0.05)]">
<div className="ai-header">
<div className="ai-icon-box">AI</div>
<div>
<div className="ai-title">AI-Generated Clinical Intelligence</div>
<div className="ai-sub">Automated insight generation · Pharma commercial decision support</div>
</div>
</div>
<div className="ai-grid">
<div className="bg-[#f2f6f5] rounded-[12px] p-[16px] border border-[#d6e0e4]">
<div className="ai-panel-title">Clinical Summary</div>
<div className="ai-panel-text" id="aiSummary">{reportData?.ai_summary}</div>
</div>
<div className="bg-[#f2f6f5] rounded-[12px] p-[16px] border border-[#d6e0e4]">
<div className="ai-panel-title">Recommended Action</div>
<div className="ai-panel-text" id="aiAction">{reportData?.ai_action}</div>
</div>
<div className="bg-[#f2f6f5] rounded-[12px] p-[16px] border border-[#d6e0e4]">
<div className="ai-panel-title">HCP Engagement</div>
<div className="ai-panel-text" id="aiHcp">{reportData?.ai_hcp}</div>
{reportData?.ai_channel && <div className="inline-block mt-[10px] bg-[#0e9f9222] border border-[#0e9f9233] text-[#087267] text-[0.68rem] font-bold py-[4px] px-[10px] rounded-[20px]" id="aiChannel">{reportData?.ai_channel}</div>}
</div>
</div>
</div>
</div>
</main>
<footer className="footer">
<div className="footer-top">
<div style={{'maxWidth': '440px'}}>
<div className="footer-brand">
<div className="logo">
<svg fill="none" height="16" viewBox="0 0 24 24" width="16" xmlns="http://www.w3.org/2000/svg">
<path d="M1 12C1 12 5 4 12 4C19 4 23 12 23 12C23 12 19 20 12 20C5 20 1 12 1 12Z" stroke="white" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" />
<circle cx="12" cy="12" r="3.5" stroke="white" strokeWidth="2" />
</svg>
</div>
<div className="text-[1.05rem] font-extrabold tracking-[-0.02em]">Netra</div>
</div>
<div className="footer-desc">Quality-gated, explainable diabetic retinopathy screening — built to catch bad images before they become bad diagnoses, and to show every lesion and confidence score behind each grade. Designed as decision support for real screening programs, not a replacement for an ophthalmologist.</div>
</div>
<div className="footer-links">
<div className="footer-links-col">
<div className="flabel">Project</div>
<a onClick={() => navTo('home')} style={{'cursor': 'pointer'}}>Home</a>
<a onClick={() => navTo('howItWorks')} style={{'cursor': 'pointer'}}>How It Works</a>
<a onClick={() => navTo('capabilities')} style={{'cursor': 'pointer'}}>Capabilities</a>
</div>
<div className="footer-links-col">
<div className="flabel">References</div>
<a href="https://www.kaggle.com/competitions/aptos2019-blindness-detection" target="_blank">APTOS 2019 Dataset</a>
</div>
</div>
</div>
<div className="footer-bottom">
<div>Built by <b>Team Netra</b></div>
<div>© 2026 Netra · Built on EfficientNetB3, trained on APTOS 2019</div>
</div>
<div className="footer-disclaimer">NETRA · AI-ASSISTED SCREENING DECISION SUPPORT · NOT A DIAGNOSIS · FOR DEMONSTRATION &amp; EDUCATIONAL PURPOSES ONLY</div>
</footer>



    </div>
  );
}
