import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getPatientRecord, createVisit, uploadScreening, saveReview } from '../api/client';
import StatusBadge from '../components/StatusBadge';
import UploadArea from '../components/UploadArea';
import EvidencePanel from '../components/EvidencePanel';

const CLASS_LABELS = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR'];

export default function PatientRecord() {
  const { patientId } = useParams();
  const [patient, setPatient] = useState(null);
  const [visits, setVisits] = useState([]);
  const [loading, setLoading] = useState(true);

  // Workspace state
  const [currentVisitId, setCurrentVisitId] = useState(null);
  const [currentEye, setCurrentEye] = useState('Left');
  const [screenings, setScreenings] = useState({ Left: null, Right: null });
  const [uploading, setUploading] = useState(false);

  // Review state
  const [grade, setGrade] = useState('');
  const [overrideReason, setOverrideReason] = useState('');
  const [clinicalNotes, setClinicalNotes] = useState('');
  const [referral, setReferral] = useState('Routine Follow-up');
  const [destCentre, setDestCentre] = useState('');
  const [followupStatus, setFollowupStatus] = useState('Not scheduled');

  useEffect(() => { load(); }, [patientId]);

  async function load() {
    try {
      setLoading(true);
      const data = await getPatientRecord(patientId);
      setPatient(data.patient);
      setVisits(data.visits);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  async function handleNewScreening() {
    try {
      const result = await createVisit(patientId);
      setCurrentVisitId(result.visit_id);
      setScreenings({ Left: null, Right: null });
      setGrade('');
      setOverrideReason('');
      setClinicalNotes('');
    } catch (e) {
      alert('Failed to create visit');
    }
  }

  async function handleUpload(file) {
    if (!currentVisitId) return;
    setUploading(true);
    try {
      const result = await uploadScreening(currentVisitId, currentEye, file);
      setScreenings((prev) => ({ ...prev, [currentEye]: result }));
    } catch (e) {
      alert('Upload failed');
    } finally {
      setUploading(false);
    }
  }

  async function handleFinalize() {
    if (!currentVisitId) return;
    const payload = {
      referral_status: referral,
      destination_centre: destCentre,
      followup_status: followupStatus,
      screenings: {},
    };
    if (grade) {
      payload.screenings[currentEye] = {
        clinician_grade: grade,
        override_reason: overrideReason,
        clinician_notes: clinicalNotes,
      };
    }
    try {
      await saveReview(currentVisitId, payload);
      alert('Review finalized.');
      load();
      setCurrentVisitId(null);
    } catch (e) {
      alert('Error saving review');
    }
  }

  const screening = screenings[currentEye];
  const isRejected = screening?.status === 'rejected';
  const isSuccess = screening?.status === 'success';
  const aiGrade = screening?.ai_grade;
  const needsOverride = grade !== '' && grade !== 'ungradable' && parseInt(grade) !== aiGrade;

  if (loading) return <div className="text-center py-20 text-muted">Loading...</div>;
  if (!patient) return <div className="text-center py-20 text-red-500">Patient not found.</div>;

  const latestVisit = visits[0];
  const reviewStatus = latestVisit ? latestVisit.review_status : 'Not screened';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Patient Header */}
      <div className="bg-white border border-border rounded-xl p-6 shadow-sm flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-full bg-sky-100 text-sky-700 text-2xl font-bold flex items-center justify-center shrink-0">
            {patient.name[0].toUpperCase()}
          </div>
          <div>
            <h1 className="text-2xl font-serif font-semibold text-heading">{patient.name}</h1>
            <div className="flex flex-wrap items-center gap-3 text-sm text-muted mt-1">
              <span className="font-mono">{patient.patient_id}</span>
              <span>·</span>
              <span>{patient.age || patient.dob || '-'} yrs</span>
              <span>·</span>
              <span>{patient.centre}</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge status={reviewStatus} />
          <button
            onClick={handleNewScreening}
            className="inline-flex items-center gap-2 bg-primary hover:bg-primary-dark text-white font-semibold px-4 py-2 rounded-lg transition-colors text-sm"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
            New Screening
          </button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {[
          { label: 'Last Screening', value: latestVisit ? latestVisit.date?.slice(0, 10) : 'None' },
          { label: 'Review Status', value: reviewStatus },
          { label: 'Follow-up', value: latestVisit?.followup_status || '-' },
          { label: 'Diabetes', value: `${patient.diabetes_type} (${patient.diabetes_duration})` },
        ].map((card) => (
          <div key={card.label} className="bg-white border border-border rounded-xl p-4">
            <div className="text-xs font-semibold text-muted uppercase tracking-wider mb-1">{card.label}</div>
            <div className="text-lg font-semibold text-heading">{card.value}</div>
          </div>
        ))}
      </div>

      {/* Screening Workspace */}
      {currentVisitId && (
        <div className="grid grid-cols-1 lg:grid-cols-[2.5fr_1fr] gap-6 mb-8">
          {/* Main Panel */}
          <div className="bg-white border border-border rounded-xl overflow-hidden shadow-sm">
            {/* Eye Tabs */}
            <div className="flex border-b border-border bg-slate-50">
              {['Left', 'Right'].map((eye) => (
                <button
                  key={eye}
                  onClick={() => setCurrentEye(eye)}
                  className={`flex-1 py-3 text-center font-semibold text-sm transition-colors border-b-2 ${
                    currentEye === eye
                      ? 'text-primary border-primary bg-white'
                      : 'text-muted border-transparent hover:text-heading'
                  }`}
                >
                  {eye} Eye
                </button>
              ))}
            </div>

            <div className="p-6">
              {/* Upload / Loading */}
              {!screening && <UploadArea onFileSelected={handleUpload} loading={uploading} />}

              {/* Rejected */}
              {isRejected && (
                <div className="mb-4">
                  <div className="flex items-start gap-3 bg-red-50 border border-red-200 text-red-800 rounded-lg p-4 mb-4">
                    <svg className="shrink-0 mt-0.5" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                    <div>
                      <div className="font-semibold">Recapture Required</div>
                      <div className="text-sm mt-0.5">{screening.quality_reason}</div>
                    </div>
                  </div>
                  <UploadArea onFileSelected={handleUpload} loading={uploading} />
                </div>
              )}

              {/* Success Results */}
              {isSuccess && (
                <>
                  <div className="flex items-start gap-3 bg-green-50 border border-green-200 text-green-800 rounded-lg p-4 mb-5">
                    <svg className="shrink-0 mt-0.5" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
                    </svg>
                    <div>
                      <div className="font-semibold">AI Analysis Complete</div>
                      <div className="text-sm mt-0.5">Quality passed · Grade: {screening.ai_label}</div>
                    </div>
                  </div>

                  <EvidencePanel
                    original={screening.original_b64}
                    gradcam={screening.gradcam_b64}
                    lesions={screening.lesion_b64}
                    lesionCount={screening.lesion_count}
                  />

                  <div className="mt-5 bg-slate-50 border border-border rounded-lg p-4">
                    <div className="text-xs font-semibold text-muted uppercase tracking-wider mb-1">AI Suggested Grade</div>
                    <div className="text-2xl font-serif font-semibold text-heading">{screening.ai_label}</div>
                    <div className="text-sm text-muted mt-1">{screening.ai_summary}</div>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Side Panel */}
          <div className="space-y-5">
            {/* Doctor Review */}
            <div className="bg-white border border-border rounded-xl overflow-hidden shadow-sm">
              <div className="px-5 py-3 bg-slate-50 border-b border-border">
                <h3 className="font-semibold text-heading">Doctor Review</h3>
              </div>
              <div className="p-5 space-y-4">
                <div>
                  <label className="block text-sm font-semibold text-heading mb-1.5">
                    Clinician Assessment ({currentEye} Eye)
                  </label>
                  <select
                    value={grade}
                    onChange={(e) => setGrade(e.target.value)}
                    className="w-full px-3 py-2.5 border border-border rounded-lg text-sm bg-white focus:outline-none focus:border-primary"
                  >
                    <option value="">-- Confirm or Override AI --</option>
                    {CLASS_LABELS.map((l, i) => (<option key={i} value={i}>{l}</option>))}
                    <option value="ungradable">Ungradable / Recapture</option>
                  </select>
                </div>

                {needsOverride && (
                  <div>
                    <label className="block text-sm font-semibold text-heading mb-1.5">
                      Reason for Override <span className="text-red-500">*</span>
                    </label>
                    <textarea
                      rows="2"
                      placeholder="Explain why the AI grade is being changed..."
                      value={overrideReason}
                      onChange={(e) => setOverrideReason(e.target.value)}
                      className="w-full px-3 py-2.5 border border-border rounded-lg text-sm focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10"
                    />
                  </div>
                )}

                <div>
                  <label className="block text-sm font-semibold text-heading mb-1.5">Clinical Notes</label>
                  <textarea
                    rows="2"
                    placeholder="Specific findings..."
                    value={clinicalNotes}
                    onChange={(e) => setClinicalNotes(e.target.value)}
                    className="w-full px-3 py-2.5 border border-border rounded-lg text-sm focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10"
                  />
                </div>
              </div>
            </div>

            {/* Referral */}
            <div className="bg-white border border-border rounded-xl overflow-hidden shadow-sm">
              <div className="px-5 py-3 bg-slate-50 border-b border-border">
                <h3 className="font-semibold text-heading">Referral & Follow-up</h3>
              </div>
              <div className="p-5 space-y-4">
                {isSuccess && (
                  <div>
                    <label className="block text-xs font-semibold text-muted uppercase mb-1">AI Suggestion</label>
                    <div className="text-sm bg-slate-50 p-3 rounded-lg text-muted">{screening.ai_action}</div>
                  </div>
                )}
                <div>
                  <label className="block text-sm font-semibold text-heading mb-1.5">Referral Decision</label>
                  <select value={referral} onChange={(e) => setReferral(e.target.value)}
                    className="w-full px-3 py-2.5 border border-border rounded-lg text-sm bg-white focus:outline-none focus:border-primary">
                    <option>Routine Follow-up</option>
                    <option>PCP Review</option>
                    <option>Specialist</option>
                    <option>Urgent</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-semibold text-heading mb-1.5">Destination Centre</label>
                  <input type="text" placeholder="e.g. Retina Clinic" value={destCentre}
                    onChange={(e) => setDestCentre(e.target.value)}
                    className="w-full px-3 py-2.5 border border-border rounded-lg text-sm focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10" />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-heading mb-1.5">Follow-up Status</label>
                  <select value={followupStatus} onChange={(e) => setFollowupStatus(e.target.value)}
                    className="w-full px-3 py-2.5 border border-border rounded-lg text-sm bg-white focus:outline-none focus:border-primary">
                    <option>Not scheduled</option>
                    <option>Scheduled</option>
                    <option>Completed</option>
                  </select>
                </div>

                <button
                  onClick={handleFinalize}
                  className="w-full bg-primary hover:bg-primary-dark text-white font-semibold py-2.5 rounded-lg transition-colors mt-2"
                >
                  Finalize Review
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Screening History */}
      <h2 className="text-2xl font-serif font-semibold text-heading mt-10 mb-4">Screening History</h2>
      <div className="space-y-3">
        {visits.length === 0 ? (
          <div className="bg-white border border-border rounded-xl text-center py-10 text-muted">
            No past screening visits found.
          </div>
        ) : (
          visits.map((v) => (
            <div key={v.visit_id} className="bg-white border border-border rounded-xl px-5 py-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div>
                <div className="font-semibold text-heading">
                  {v.date?.slice(0, 10)} · Visit {v.visit_id.slice(-6)}
                </div>
                <div className="text-sm text-muted mt-0.5">
                  Status: {v.review_status} · Follow-up: {v.followup_status || 'Pending'}
                </div>
              </div>
              <button
                onClick={() => {
                  setCurrentVisitId(v.visit_id);
                  // Load existing screenings for this visit
                  const s = { Left: null, Right: null };
                  if (v.screenings) {
                    for (const [eye, data] of Object.entries(v.screenings)) {
                      s[eye] = {
                        status: data.quality_pass ? 'success' : 'rejected',
                        quality_reason: data.quality_reason,
                        ai_grade: data.ai_grade,
                        ai_label: CLASS_LABELS[data.ai_grade] || 'N/A',
                        ai_summary: '',
                        ai_action: '',
                        original_b64: data.original_b64_path,
                        gradcam_b64: data.gradcam_b64_path,
                        lesion_b64: data.lesion_overlay_b64_path,
                        lesion_count: data.lesion_count,
                      };
                    }
                  }
                  setScreenings(s);
                }}
                className="px-4 py-2 border border-border rounded-lg text-sm font-semibold text-heading hover:bg-slate-50 transition-colors"
              >
                Open Record
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
