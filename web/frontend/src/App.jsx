import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Analyze from './pages/Analyze'
import Research from './pages/Research'
import Pipeline from './pages/Pipeline'
import './index.css'

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/"          element={<Home />} />
        <Route path="/analyze"   element={<Analyze />} />
        <Route path="/research"  element={<Research />} />
        <Route path="/pipeline"  element={<Pipeline />} />
      </Routes>
    </BrowserRouter>
  )
}
