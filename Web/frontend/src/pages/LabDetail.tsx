import React from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getLabByCode } from '../data/labs'

const LabDetail: React.FC = () => {
  const { code } = useParams<{ code: string }>()
  const navigate = useNavigate()
  const lab = code ? getLabByCode(code) : null

  if (!lab) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-4">Lab Not Found</h1>
          <button
            onClick={() => navigate('/labs')}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Back to Labs
          </button>
        </div>
      </div>
    )
  }

  const phaseColor = 
    lab.phase === 1 ? 'blue' :
    lab.phase === 2 ? 'green' :
    'purple'

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      {/* Navigation */}
      <nav className="bg-slate-800 border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-white">Lab Platform</h1>
          <div className="flex gap-3">
            <button
              onClick={() => navigate('/labs')}
              className="px-4 py-2 bg-slate-700 text-white rounded hover:bg-slate-600 transition"
            >
              ← Back to Labs
            </button>
          </div>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className={`text-sm font-bold text-${phaseColor}-400`}>Lab {lab.code} • Phase {lab.phase}</div>
            <span className={`bg-${lab.type === 'mandatory' ? 'blue' : 'green'}-600 text-white text-xs px-3 py-1 rounded-full font-semibold`}>
              {lab.type.charAt(0).toUpperCase() + lab.type.slice(1)}
            </span>
          </div>
          <h1 className="text-4xl font-bold text-white mb-4">{lab.title}</h1>
          <p className="text-xl text-gray-300">{lab.description}</p>
        </div>

        {/* Key Info */}
        <div className={`bg-${phaseColor}-900 border border-${phaseColor}-500 rounded-lg p-6 mb-8`}>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-gray-400">Points</div>
              <div className={`text-3xl font-bold text-${phaseColor}-400`}>{lab.points}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Type</div>
              <div className="text-xl font-bold text-white capitalize">{lab.type}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Tasks</div>
              <div className="text-3xl font-bold text-white">{lab.tasks.length}</div>
            </div>
          </div>
          {lab.notes && (
            <div className="mt-4 pt-4 border-t border-slate-500">
              <p className="text-sm text-gray-300">📌 {lab.notes}</p>
            </div>
          )}
        </div>

        {/* Tasks */}
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-6 mb-8">
          <h2 className="text-2xl font-bold text-white mb-6">Tasks</h2>
          <div className="space-y-6">
            {lab.tasks.map((task, idx) => (
              <div key={idx} className="border-l-4 border-blue-500 pl-4">
                <h3 className="text-lg font-semibold text-white mb-2">{idx + 1}. {task.title}</h3>
                <p className="text-gray-300 mb-3">{task.description}</p>
                <div>
                  <p className="text-sm text-gray-400 mb-2">Deliverables:</p>
                  <ul className="list-disc list-inside space-y-1">
                    {task.deliverables.map((del, didx) => (
                      <li key={didx} className="text-gray-300 text-sm">✓ {del}</li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Self-Study Materials */}
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-6 mb-8">
          <h2 className="text-2xl font-bold text-white mb-4">Self-Study Materials</h2>
          <ul className="space-y-2">
            {lab.selfStudyMaterials.map((material, idx) => (
              <li key={idx} className="text-gray-300 flex items-center">
                <span className="text-blue-400 mr-3">📚</span>
                {material}
              </li>
            ))}
          </ul>
        </div>

        {/* Required Knowledge */}
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-6 mb-8">
          <h2 className="text-2xl font-bold text-white mb-4">Required Knowledge</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {lab.requiredKnowledge.map((knowledge, idx) => (
              <div key={idx} className="bg-slate-700 p-3 rounded flex items-start">
                <span className="text-green-400 mr-3">✓</span>
                <span className="text-gray-300 text-sm">{knowledge}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4">
          <button className="flex-1 px-4 py-3 bg-blue-600 text-white rounded font-semibold hover:bg-blue-700 transition">
            Start Lab
          </button>
          <button
            onClick={() => navigate('/labs')}
            className="flex-1 px-4 py-3 bg-slate-700 text-white rounded font-semibold hover:bg-slate-600 transition"
          >
            Back to Labs
          </button>
        </div>
      </div>
    </div>
  )
}

export default LabDetail
