import React, { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'

const Login: React.FC = () => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const navigate = useNavigate()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    navigate('/dashboard')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 flex items-center justify-center">
      <div className="bg-slate-800 p-8 rounded-lg border border-slate-700 w-full max-w-md">
        <h2 className="text-3xl font-bold text-white mb-6 text-center">Log In</h2>
        <form onSubmit={handleSubmit}>
          <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} className="w-full px-4 py-2 mb-4 bg-slate-700 text-white rounded" />
          <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full px-4 py-2 mb-4 bg-slate-700 text-white rounded" />
          <button type="submit" className="w-full px-4 py-2 bg-blue-600 text-white rounded font-semibold">Log In</button>
        </form>
        <p className="text-center text-gray-400 mt-4">Don't have an account? <Link to="/register" className="text-blue-400">Sign up</Link></p>
      </div>
    </div>
  )
}

export default Login
