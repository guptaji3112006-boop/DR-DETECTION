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
            <img src="/static/img/logo.png" alt="Netra Logo" className="h-16 w-auto" />
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
