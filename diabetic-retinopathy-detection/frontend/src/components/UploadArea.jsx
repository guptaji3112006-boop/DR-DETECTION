import { useRef } from 'react';

export default function UploadArea({ onFileSelected, loading }) {
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file) onFileSelected(file);
  };

  const handleChange = (e) => {
    const file = e.target.files?.[0];
    if (file) onFileSelected(file);
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-primary">
        <svg className="animate-spin h-10 w-10 mb-4" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
        </svg>
        <span className="font-semibold text-lg">Analyzing Image...</span>
        <span className="text-muted text-sm mt-1">Running IQA + AI grading</span>
      </div>
    );
  }

  return (
    <label
      className="flex flex-col items-center justify-center border-2 border-dashed border-border rounded-xl py-12 px-6 text-center cursor-pointer bg-slate-50 hover:border-primary hover:bg-primary-light/30 transition-colors"
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      <svg className="mb-4 text-primary" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="17 8 12 3 7 8"/>
        <line x1="12" y1="3" x2="12" y2="15"/>
      </svg>
      <span className="font-semibold text-heading text-lg mb-1">Click to upload fundus image</span>
      <span className="text-muted text-sm">JPEG, PNG up to 10 MB</span>
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={handleChange} />
    </label>
  );
}
