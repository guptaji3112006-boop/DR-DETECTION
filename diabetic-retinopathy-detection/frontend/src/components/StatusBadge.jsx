const styles = {
  'Review pending': 'bg-orange-50 text-orange-700 border border-orange-200',
  Reviewed: 'bg-green-50 text-green-700 border border-green-200',
  'Not screened': 'bg-slate-100 text-slate-500 border border-slate-200',
};

export default function StatusBadge({ status }) {
  const cls = styles[status] || styles['Not screened'];
  return (
    <span className={`inline-flex items-center px-3 py-0.5 rounded-full text-sm font-medium ${cls}`}>
      {status}
    </span>
  );
}
