import { useState, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { getPatients, createPatient } from '../api/client';
import StatusBadge from '../components/StatusBadge';
import Modal from '../components/Modal';

export default function Dashboard() {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [centreFilter, setCentreFilter] = useState('all');
  const [modalOpen, setModalOpen] = useState(false);
  const [formData, setFormData] = useState({
    name: '', age_or_dob: '', contact: '', centre: '',
    diabetes_type: 'Unknown', diabetes_duration: 'Unknown',
    vision_complaints: '', clinical_notes: '',
  });
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => { loadPatients(); }, []);

  async function loadPatients() {
    try {
      setLoading(true);
      const data = await getPatients();
      setPatients(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const centres = useMemo(
    () => [...new Set(patients.map((p) => p.centre).filter(Boolean))],
    [patients]
  );

  function getReviewStatus(p) {
    if (!p.visits?.length) return 'Not screened';
    return p.visits[0].review_status || 'Review pending';
  }

  function getFollowup(p) {
    if (!p.visits?.length) return '-';
    return p.visits[0].followup_status || 'Not scheduled';
  }

  const filtered = useMemo(() => {
    return patients.filter((p) => {
      const q = search.toLowerCase();
      const matchQ = p.name.toLowerCase().includes(q) || p.patient_id.toLowerCase().includes(q);
      const st = getReviewStatus(p);
      const matchStatus = statusFilter === 'all' || st === statusFilter;
      const matchCentre = centreFilter === 'all' || p.centre === centreFilter;
      return matchQ && matchStatus && matchCentre;
    });
  }, [patients, search, statusFilter, centreFilter]);

  async function handleSubmit(e) {
    e.preventDefault();
    setSubmitting(true);
    try {
      const data = { ...formData };
      if (data.age_or_dob.includes('-')) {
        data.dob = data.age_or_dob;
      } else {
        data.age = data.age_or_dob;
      }
      delete data.age_or_dob;
      const result = await createPatient(data);
      setModalOpen(false);
      window.location.href = `/patients/${result.patient_id}`;
    } catch (e) {
      alert('Registration failed');
    } finally {
      setSubmitting(false);
    }
  }

  function updateForm(field, value) {
    setFormData((prev) => ({ ...prev, [field]: value }));
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
        <h1 className="text-4xl font-serif font-semibold text-heading">Patients</h1>
        <button
          onClick={() => setModalOpen(true)}
          className="inline-flex items-center gap-2 bg-primary hover:bg-primary-dark text-white font-semibold px-5 py-2.5 rounded-lg transition-colors"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
          </svg>
          Add Patient
        </button>
      </div>

      {/* Filters */}
      <div className="bg-white border border-border rounded-xl shadow-sm p-4 flex flex-wrap gap-3 mb-6">
        <div className="relative flex-1 min-w-[250px]">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          <input
            type="text"
            placeholder="Search by name or ID..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 border border-border rounded-lg text-sm focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10"
          />
        </div>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="px-4 py-2.5 border border-border rounded-lg text-sm text-heading bg-white focus:outline-none focus:border-primary"
        >
          <option value="all">All Statuses</option>
          <option value="Review pending">Review Pending</option>
          <option value="Reviewed">Reviewed</option>
          <option value="Not screened">Not Screened</option>
        </select>
        <select
          value={centreFilter}
          onChange={(e) => setCentreFilter(e.target.value)}
          className="px-4 py-2.5 border border-border rounded-lg text-sm text-heading bg-white focus:outline-none focus:border-primary"
        >
          <option value="all">All Centres</option>
          {centres.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      {/* Content */}
      {loading ? (
        <div className="text-center py-16 text-muted">Loading patients...</div>
      ) : error ? (
        <div className="text-center py-16 text-red-500">{error}</div>
      ) : filtered.length === 0 ? (
        <div className="bg-white border border-border rounded-xl text-center py-16 text-muted">
          {patients.length === 0 ? 'No patients registered yet. Click "Add Patient" to begin.' : 'No patients match your filters.'}
        </div>
      ) : (
        <>
          {/* Desktop Table */}
          <div className="hidden md:block bg-white border border-border rounded-xl shadow-sm overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-slate-50 border-b border-border">
                  {['Patient ID', 'Name', 'Age', 'Centre', 'Status', 'Follow-up', ''].map((h) => (
                    <th key={h} className="text-left px-5 py-3 text-xs font-semibold text-muted uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((p) => (
                  <tr key={p.patient_id} className="border-b border-border last:border-0 hover:bg-slate-50 transition-colors">
                    <td className="px-5 py-4 font-mono text-sm text-muted">{p.patient_id}</td>
                    <td className="px-5 py-4 font-semibold">{p.name}</td>
                    <td className="px-5 py-4 text-sm">{p.age || p.dob || '-'}</td>
                    <td className="px-5 py-4 text-sm">{p.centre || '-'}</td>
                    <td className="px-5 py-4"><StatusBadge status={getReviewStatus(p)} /></td>
                    <td className="px-5 py-4 text-sm">{getFollowup(p)}</td>
                    <td className="px-5 py-4">
                      <Link to={`/patients/${p.patient_id}`} className="text-primary font-semibold hover:underline inline-flex items-center gap-1 text-sm">
                        View Record
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Mobile Cards */}
          <div className="md:hidden flex flex-col gap-3">
            {filtered.map((p) => (
              <div key={p.patient_id} className="bg-white border border-border rounded-xl p-4">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <div className="text-xs text-muted font-mono">{p.patient_id}</div>
                    <div className="font-bold text-heading text-lg">{p.name}</div>
                  </div>
                  <StatusBadge status={getReviewStatus(p)} />
                </div>
                <div className="text-sm text-muted mb-3">
                  Age: {p.age || p.dob || '-'} · Centre: {p.centre || '-'}
                </div>
                <Link
                  to={`/patients/${p.patient_id}`}
                  className="block text-center bg-white border border-border rounded-lg py-2 text-sm font-semibold text-heading hover:bg-slate-50 transition-colors"
                >
                  View Record
                </Link>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Registration Modal */}
      <Modal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        title="Add New Patient"
        footer={
          <>
            <button onClick={() => setModalOpen(false)} className="px-4 py-2 border border-border rounded-lg font-semibold text-heading hover:bg-slate-50 transition-colors">
              Cancel
            </button>
            <button onClick={handleSubmit} disabled={submitting} className="px-5 py-2 bg-primary hover:bg-primary-dark text-white font-semibold rounded-lg transition-colors disabled:opacity-50">
              {submitting ? 'Saving...' : 'Register Patient'}
            </button>
          </>
        }
      >
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <div>
              <label className="block text-sm font-semibold text-heading mb-1.5">
                Full Name <span className="text-red-500">*</span>
              </label>
              <input required type="text" placeholder="e.g. Rahul Sharma" value={formData.name}
                onChange={(e) => updateForm('name', e.target.value)}
                className="w-full px-3 py-2.5 border border-border rounded-lg text-sm focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10" />
            </div>
            <div>
              <label className="block text-sm font-semibold text-heading mb-1.5">Contact</label>
              <input type="text" placeholder="Optional" value={formData.contact}
                onChange={(e) => updateForm('contact', e.target.value)}
                className="w-full px-3 py-2.5 border border-border rounded-lg text-sm focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10" />
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <div>
              <label className="block text-sm font-semibold text-heading mb-1.5">
                Age or DOB <span className="text-red-500">*</span>
              </label>
              <input required type="text" placeholder="Age or YYYY-MM-DD" value={formData.age_or_dob}
                onChange={(e) => updateForm('age_or_dob', e.target.value)}
                className="w-full px-3 py-2.5 border border-border rounded-lg text-sm focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10" />
            </div>
            <div>
              <label className="block text-sm font-semibold text-heading mb-1.5">
                Screening Centre <span className="text-red-500">*</span>
              </label>
              <input required type="text" placeholder="e.g. Main Hospital" value={formData.centre}
                onChange={(e) => updateForm('centre', e.target.value)}
                className="w-full px-3 py-2.5 border border-border rounded-lg text-sm focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10" />
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <div>
              <label className="block text-sm font-semibold text-heading mb-1.5">Diabetes Type</label>
              <select value={formData.diabetes_type} onChange={(e) => updateForm('diabetes_type', e.target.value)}
                className="w-full px-3 py-2.5 border border-border rounded-lg text-sm bg-white focus:outline-none focus:border-primary">
                <option>Unknown</option><option>Type 1</option><option>Type 2</option><option>Other</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-semibold text-heading mb-1.5">Diabetes Duration</label>
              <select value={formData.diabetes_duration} onChange={(e) => updateForm('diabetes_duration', e.target.value)}
                className="w-full px-3 py-2.5 border border-border rounded-lg text-sm bg-white focus:outline-none focus:border-primary">
                <option>Unknown</option><option>{'< 5 Years'}</option><option>5-10 Years</option><option>{'> 10 Years'}</option>
              </select>
            </div>
          </div>
          <div>
            <label className="block text-sm font-semibold text-heading mb-1.5">Vision Complaints</label>
            <textarea rows="2" placeholder="e.g. Blurry vision, floaters..." value={formData.vision_complaints}
              onChange={(e) => updateForm('vision_complaints', e.target.value)}
              className="w-full px-3 py-2.5 border border-border rounded-lg text-sm focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10" />
          </div>
          <div>
            <label className="block text-sm font-semibold text-heading mb-1.5">Clinical Notes</label>
            <textarea rows="2" placeholder="Relevant medical history..." value={formData.clinical_notes}
              onChange={(e) => updateForm('clinical_notes', e.target.value)}
              className="w-full px-3 py-2.5 border border-border rounded-lg text-sm focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10" />
          </div>
        </form>
      </Modal>
    </div>
  );
}
