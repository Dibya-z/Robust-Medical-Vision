import { NavLink } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav className="navbar">
      <NavLink to="/" className="navbar-brand">
        <div className="navbar-logo">🔬</div>
        <span>DermaSense <span style={{ color: 'var(--teal)' }}>AI</span></span>
      </NavLink>

      <div className="navbar-links">
        <NavLink to="/"         className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} end>Home</NavLink>
        <NavLink to="/analyze"  className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Analyze</NavLink>
        <NavLink to="/research" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Research</NavLink>
        <NavLink to="/pipeline" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Pipeline</NavLink>
      </div>
    </nav>
  )
}
