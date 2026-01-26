import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Landing from './pages/Landing'
import Login from './pages/Login'
import Register from './pages/Register'
import Dashboard from './pages/Dashboard'
import Labs from './pages/Labs'
import LabDetail from './pages/LabDetail'
import LabTree from './pages/LabTree'
import './App.css'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/labs" element={<Labs />} />
        <Route path="/labs/:code" element={<LabDetail />} />
        <Route path="/tree" element={<LabTree />} />
      </Routes>
    </Router>
  )
}

export default App
