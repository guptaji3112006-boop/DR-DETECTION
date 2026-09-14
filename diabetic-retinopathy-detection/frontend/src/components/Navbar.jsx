import { Link, useLocation } from 'react-router-dom';
import { useState } from 'react';

export default function Navbar() {
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);

  const linkClass = (path) =>
    `transition-colors font-medium ${
      location.pathname.startsWith(path)
        ? 'text-heading font-semibold'
        : 'text-muted hover:text-heading'
    }`;

  return (
    <header className="sticky top-0 z-50 bg-white border-b border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <a href="/" className="flex items-center gap-3 no-underline">
            <div className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M1 12C1 12 5 4 12 4C19 4 23 12 23 12C23 12 19 20 12 20C5 20 1 12 1 12Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <circle cx="12" cy="12" r="3.5" stroke="white" strokeWidth="2"/>
              </svg>
            </div>
            <span className="text-xl font-extrabold text-heading tracking-tight">Netra</span>
          </a>

          {/* Desktop Nav */}
          <nav className="hidden md:flex items-center gap-6 ml-8">
            <a href="/" className={linkClass('__home__')}>Home</a>
            <Link to="/patients" className={linkClass('/patients')}>Patients</Link>
          </nav>
        </div>

        {/* Mobile Hamburger */}
        <button
          className="md:hidden p-2 text-muted"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 6h16M4 12h16M4 18h16" strokeLinecap="round"/>
          </svg>
        </button>
      </div>

      {/* Mobile Menu */}
      {menuOpen && (
        <nav className="md:hidden border-t border-border bg-white px-4 py-3 flex flex-col gap-3">
          <a href="/" className="text-muted font-medium hover:text-heading">Home</a>
          <Link to="/patients" className="text-muted font-medium hover:text-heading" onClick={() => setMenuOpen(false)}>
            Patients
          </Link>
        </nav>
      )}
    </header>
  );
}
