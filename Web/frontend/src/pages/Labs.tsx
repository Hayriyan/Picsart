import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getLabsByPhase, getMandatoryLabsPoints, getOptionalLabsPoints } from '../data/labs'
import type { Lab } from '../data/labs'

const Labs: React.FC = () => {
  const navigate = useNavigate()
  const [selectedPhase, setSelectedPhase] = useState<1 | 2 | 3>(1)
  const [searchTerm, setSearchTerm] = useState('')
  const [filterType, setFilterType] = useState<'all' | 'mandatory' | 'optional'>('all')

  const labs = getLabsByPhase(selectedPhase)
  const filteredLabs = labs.filter(lab => {
    const matchesSearch = lab.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         lab.code.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesType = filterType === 'all' || lab.type === filterType
    return matchesSearch && matchesType
  })

  const mandatoryPoints = getMandatoryLabsPoints(selectedPhase)
  const optionalPoints = getOptionalLabsPoints(selectedPhase)

  const phaseNames: Record<1 | 2 | 3, string> = {
    1: 'Computational & Programming Foundations',
    2: 'Machine Learning Deep Dive',
    3: 'Deep Learning & Data Science'
  }

  const phaseColors: Record<1 | 2 | 3, string> = {
    1: 'blue',
    2: 'green',
    3: 'purple'
  }

  const color = phaseColors[selectedPhase]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      {/* Navigation */}
      <nav className="bg-slate-800 border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-white">Lab Platform</h1>
          <div className="flex gap-3">
            <button
              onClick={() => navigate('/tree')}
              className="px-4 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700 transition"
            >
              Tree View
            </button>
            <button
              onClick={() => navigate('/dashboard')}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
            >
              Dashboard
            </button>
            <button
              onClick={() => {
                localStorage.removeItem('token')
                navigate('/')
              }}
              className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition"
            >
              Logout
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Phase Selection */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-4">Select a Phase</h2>
          <div className="flex gap-4 flex-wrap">
            {[1, 2, 3].map(phase => (
              <button
                key={phase}
                onClick={() => setSelectedPhase(phase as 1 | 2 | 3)}
                className={`px-6 py-2 rounded-lg font-semibold transition ${
                  selectedPhase === phase
                    ? phase === 1 ? 'bg-blue-600 text-white' : phase === 2 ? 'bg-green-600 text-white' : 'bg-purple-600 text-white'
                    : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                }`}
              >
                Phase {phase}
              </button>
            ))}
          </div>
        </div>

        {/* Phase Info Card */}
        <div className={`bg-slate-800 rounded-lg border-2 mb-8 p-6 ${
          color === 'blue' ? 'border-blue-500' : color === 'green' ? 'border-green-500' : 'border-purple-500'
        }`}>
          <h3 className="text-xl font-bold text-white mb-2">Phase {selectedPhase}: {phaseNames[selectedPhase]}</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className={`text-sm ${color === 'blue' ? 'text-blue-400' : color === 'green' ? 'text-green-400' : 'text-purple-400'} font-semibold mb-1`}>Mandatory Labs</div>
              <div className="text-2xl font-bold text-white">{mandatoryPoints} points</div>
            </div>
            <div>
              <div className={`text-sm ${color === 'blue' ? 'text-blue-400' : color === 'green' ? 'text-green-400' : 'text-purple-400'} font-semibold mb-1`}>Optional Labs</div>
              <div className="text-2xl font-bold text-white">{optionalPoints} points</div>
            </div>
          </div>
        </div>

        {/* Search and Filter */}
        <div className="mb-8 grid grid-cols-1 md:grid-cols-2 gap-4">
          <input
            type="text"
            placeholder="Search labs by title or code..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="px-4 py-2 bg-slate-700 text-white placeholder-gray-400 rounded border border-slate-600 focus:outline-none focus:border-blue-500"
          />
          <div className="flex gap-2">
            {['all', 'mandatory', 'optional'].map(type => (
              <button
                key={type}
                onClick={() => setFilterType(type as 'all' | 'mandatory' | 'optional')}
                className={`px-4 py-2 rounded transition capitalize ${
                  filterType === type
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        {/* Labs Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {filteredLabs.map(lab => (
            <div
              key={lab.id}
              className="bg-slate-800 rounded-lg border border-slate-700 p-6 hover:border-blue-500 transition"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="text-sm text-gray-400">{lab.code}</div>
                  <h3 className="text-lg font-bold text-white">{lab.title}</h3>
                </div>
                <span className={`px-3 py-1 rounded text-xs font-semibold whitespace-nowrap ${
                  lab.type === 'mandatory'
                    ? 'bg-red-900 text-red-200'
                    : 'bg-gray-700 text-gray-200'
                }`}>
                  {lab.type === 'mandatory' ? 'Required' : 'Optional'}
                </span>
              </div>

              <p className="text-gray-400 text-sm mb-4">{lab.description}</p>

              <div className="flex items-center justify-between mb-4">
                <div className="flex gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Points: </span>
                    <span className="text-blue-400 font-semibold">{lab.points}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Tasks: </span>
                    <span className="text-blue-400 font-semibold">{lab.tasks.length}</span>
                  </div>
                </div>
              </div>

              <div className="mb-4">
                <div className="text-xs text-gray-400 mb-2">Required Knowledge:</div>
                <div className="flex flex-wrap gap-2">
                  {lab.requiredKnowledge.slice(0, 3).map((skill, idx) => (
                    <span key={idx} className="bg-slate-700 text-gray-300 text-xs px-2 py-1 rounded">
                      {skill}
                    </span>
                  ))}
                  {lab.requiredKnowledge.length > 3 && (
                    <span className="bg-slate-700 text-gray-300 text-xs px-2 py-1 rounded">
                      +{lab.requiredKnowledge.length - 3}
                    </span>
                  )}
                </div>
              </div>

              <button
                onClick={() => navigate(`/labs/${lab.code}`)}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition text-sm font-semibold"
              >
                View Details
              </button>
            </div>
          ))}
        </div>

        {filteredLabs.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-400 text-lg">No labs found matching your criteria</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default Labs
