import React from 'react'
import { Link } from 'react-router-dom'

const Landing: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      <nav className="flex justify-between items-center p-6 max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-white">LabTree</h1>
        <div className="flex gap-4">
          <Link to="/login" className="px-6 py-2 bg-blue-600 text-white rounded-lg">Log In</Link>
          <Link to="/register" className="px-6 py-2 border-2 border-blue-600 text-blue-400 rounded-lg">Sign Up</Link>
        </div>
      </nav>
      <div className="max-w-7xl mx-auto px-6 py-20 text-center">
        <h2 className="text-5xl font-bold text-white mb-4">Master AI & ML Through Structured Labs</h2>
        <p className="text-xl text-gray-400 mb-8">A 15-month program with 32+ comprehensive labs</p>
        <Link to="/login" className="px-8 py-3 bg-blue-600 text-white rounded-lg font-semibold">Get Started</Link>
      </div>
    </div>
  )
}

export default Landing
