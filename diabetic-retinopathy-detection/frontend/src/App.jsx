import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import PatientRecord from './pages/PatientRecord';

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/patients" element={<Dashboard />} />
        <Route path="/patients/:patientId" element={<PatientRecord />} />
        <Route path="*" element={<Navigate to="/patients" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
