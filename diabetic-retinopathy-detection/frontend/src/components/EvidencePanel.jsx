import { useState } from 'react';

function Panel({ src, label, alt, unavailableText }) {
  const [enlarged, setEnlarged] = useState(false);

  return (
    <>
      <div
        className="relative bg-slate-900 rounded-lg overflow-hidden aspect-square flex items-center justify-center cursor-pointer group"
        onClick={() => src && setEnlarged(true)}
      >
        {src ? (
          <img
            src={`data:image/png;base64,${src}`}
            alt={alt}
            className="w-full h-full object-contain"
          />
        ) : (
          <span className="text-slate-500 text-sm text-center px-4">
            {unavailableText || 'Unavailable'}
          </span>
        )}
        <div className="absolute bottom-0 inset-x-0 bg-slate-900/80 backdrop-blur text-white text-xs text-center py-1.5 px-2">
          {label}
        </div>
        {src && (
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
            <svg className="text-white opacity-0 group-hover:opacity-80 transition-opacity" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
            </svg>
          </div>
        )}
      </div>

      {/* Lightbox */}
      {enlarged && (
        <div
          className="fixed inset-0 z-[200] bg-black/80 flex items-center justify-center p-8 cursor-pointer"
          onClick={() => setEnlarged(false)}
        >
          <img
            src={`data:image/png;base64,${src}`}
            alt={alt}
            className="max-w-full max-h-full object-contain rounded-lg"
          />
        </div>
      )}
    </>
  );
}

export default function EvidencePanel({ original, gradcam, lesions, lesionCount }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
      <Panel src={original} label="Original Image" alt="Original fundus" unavailableText="No image" />
      <Panel src={gradcam} label="Grad-CAM Evidence" alt="Grad-CAM heatmap" unavailableText="Grad-CAM unavailable" />
      <Panel
        src={lesions}
        label={`Lesion Candidates${lesionCount >= 0 ? `: ${lesionCount}` : ''}`}
        alt="Lesion overlay"
        unavailableText="Lesion detection unavailable"
      />
    </div>
  );
}
