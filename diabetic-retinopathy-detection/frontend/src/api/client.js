const API_BASE = '';

export async function getPatients() {
  const res = await fetch(`${API_BASE}/api/patients`);
  if (!res.ok) throw new Error('Failed to fetch patients');
  const data = await res.json();
  return data.patients;
}

export async function createPatient(patientData) {
  const res = await fetch(`${API_BASE}/api/patients`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patientData),
  });
  if (!res.ok) throw new Error('Failed to create patient');
  return res.json();
}

export async function getPatientRecord(patientId) {
  const res = await fetch(`${API_BASE}/api/patients/${patientId}`);
  if (!res.ok) throw new Error('Failed to fetch patient record');
  return res.json();
}

export async function createVisit(patientId) {
  const res = await fetch(`${API_BASE}/api/visits`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ patient_id: patientId }),
  });
  if (!res.ok) throw new Error('Failed to create visit');
  return res.json();
}

export async function uploadScreening(visitId, eye, file) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('visit_id', visitId);
  fd.append('eye', eye);
  const res = await fetch(`${API_BASE}/api/screenings`, {
    method: 'POST',
    body: fd,
  });
  if (!res.ok) throw new Error('Screening upload failed');
  return res.json();
}

export async function saveReview(visitId, reviewData) {
  const res = await fetch(`${API_BASE}/api/visits/${visitId}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reviewData),
  });
  if (!res.ok) throw new Error('Failed to save review');
  return res.json();
}

export async function generateReport(reportData) {
  const res = await fetch(`${API_BASE}/generate_report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reportData),
  });
  if (!res.ok) throw new Error('Failed to generate report');
  return res.blob();
}
