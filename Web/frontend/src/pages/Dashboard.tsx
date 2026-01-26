import React from 'react'
import { useNavigate } from 'react-router-dom'

const Dashboard: React.FC = () => {
  const navigate = useNavigate()

  const handleLogout = () => {
    localStorage.removeItem('token')
    navigate('/')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      {/* Navigation */}
      <nav className="bg-slate-800 border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-white">Lab Platform</h1>
          <div className="flex gap-3">
            <button
              onClick={() => navigate('/tree')}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
            >
              Tree View
            </button>
            <button
              onClick={() => navigate('/labs')}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
            >
              Labs
            </button>
            <button
              onClick={handleLogout}
              className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition"
            >
              Logout
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Welcome Card */}
          <div className="md:col-span-3 bg-slate-800 rounded-lg border border-slate-700 p-6">
            <h2 className="text-2xl font-bold text-white mb-2">Welcome to the Lab Platform! 🎉</h2>
            <p className="text-gray-400">Start your 15-month ML/DL journey. View the program structure as an interactive tree or browse individual labs.</p>
          </div>

          {/* Tree View Card */}
          <div
            onClick={() => navigate('/tree')}
            className="bg-slate-800 rounded-lg border border-slate-700 p-6 hover:border-yellow-500 transition cursor-pointer"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white">🌳 Tree View</h3>
            </div>
            <p className="text-gray-400 mb-4">Visualize the complete 15-month program structure with all phases and labs as an interactive tree.</p>
            <p className="text-yellow-400 font-semibold">View Program Structure</p>
          </div>

          {/* Phase 1 Card */}
          <div
            onClick={() => navigate('/labs')}
            className="bg-slate-800 rounded-lg border border-slate-700 p-6 hover:border-blue-500 transition cursor-pointer"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white">Phase 1</h3>
              <span className="bg-blue-600 text-white px-3 py-1 rounded text-sm">Intro</span>
            </div>
            <p className="text-gray-400 mb-4">Computational & Programming Foundations</p>
            <p className="text-blue-400 font-semibold">9 Mandatory + 2 Optional Labs</p>
          </div>

          {/* Phase 2 Card */}
          <div
            onClick={() => navigate('/labs')}
            className="bg-slate-800 rounded-lg border border-slate-700 p-6 hover:border-green-500 transition cursor-pointer"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white">Phase 2</h3>
              <span className="bg-green-600 text-white px-3 py-1 rounded text-sm">Intermediate</span>
            </div>
            <p className="text-gray-400 mb-4">Machine Learning Deep Dive</p>
            <p className="text-green-400 font-semibold">8 Mandatory + 1 Optional Lab</p>
          </div>

          {/* Phase 3 Card */}
          <div
            onClick={() => navigate('/labs')}
            className="bg-slate-800 rounded-lg border border-slate-700 p-6 hover:border-purple-500 transition cursor-pointer"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white">Phase 3</h3>
              <span className="bg-purple-600 text-white px-3 py-1 rounded text-sm">Advanced</span>
            </div>
            <p className="text-gray-400 mb-4">Deep Learning & Data Science</p>
            <p className="text-purple-400 font-semibold">8 Mandatory + 2 Optional Labs</p>
          </div>
        </div>

        {/* Info Cards */}
        <div className="mt-12">
          <h3 className="text-xl font-bold text-white mb-6">Progression Requirements</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
              <div className="text-sm text-gray-400 mb-2">Phase 1</div>
              <div className="text-2xl font-bold text-blue-400">100 pts</div>
              <div className="text-xs text-gray-500 mt-2">Minimum to pass and advance to Phase 2</div>
            </div>
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
              <div className="text-sm text-gray-400 mb-2">Phase 2</div>
              <div className="text-2xl font-bold text-green-400">100 pts</div>
              <div className="text-xs text-gray-500 mt-2">Minimum to pass and advance to Phase 3</div>
            </div>
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
              <div className="text-sm text-gray-400 mb-2">Phase 3</div>
              <div className="text-2xl font-bold text-purple-400">100 pts</div>
              <div className="text-xs text-gray-500 mt-2">Minimum to complete program</div>
            </div>
          </div>
        </div>

        {/* Statistics */}
        <div className="mt-12">
          <h3 className="text-xl font-bold text-white mb-6">Your Progress</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
              <div className="text-3xl font-bold text-blue-400">0</div>
              <div className="text-gray-400 mt-2">Labs Completed</div>
            </div>
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
              <div className="text-3xl font-bold text-green-400">0</div>
              <div className="text-gray-400 mt-2">Points Earned</div>
            </div>
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
              <div className="text-3xl font-bold text-purple-400">0%</div>
              <div className="text-gray-400 mt-2">Program Progress</div>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="mt-12 bg-gradient-to-r from-blue-900 to-purple-900 rounded-lg border border-blue-500 p-8 text-center">
          <h3 className="text-2xl font-bold text-white mb-2">Ready to start learning?</h3>
          <p className="text-gray-300 mb-6">Explore all labs, understand requirements, and begin your AI/ML journey.</p>
          <button
            onClick={() => navigate('/labs')}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition"
          >
            Browse All Labs →
          </button>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
