import React, { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import * as d3 from 'd3'
import { labs } from '../data/labs'

interface TreeNode {
  id: string
  code: string
  name: string
  phase: number
  points: number
  type: string
  parent?: string
  children?: TreeNode[]
  url?: string
}

const LabTree: React.FC = () => {
  const navigate = useNavigate()
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current) return

    // Build tree data structure
    const root: TreeNode = {
      id: 'root',
      code: 'ROOT',
      name: '15-Month ML/DL Program',
      phase: 0,
      points: 300,
      type: 'root'
    }

    // Phase 1 as root children
    const phase1Labs = labs.filter(lab => lab.phase === 1).slice(0, 5)
    const phase2Labs = labs.filter(lab => lab.phase === 2).slice(0, 5)
    const phase3Labs = labs.filter(lab => lab.phase === 3).slice(0, 5)

    root.children = [
      {
        id: 'phase1',
        code: 'P1',
        name: 'Phase 1: Foundations',
        phase: 1,
        points: 127,
        type: 'phase',
        parent: 'root',
        children: phase1Labs.map(lab => ({
          id: lab.id,
          code: lab.code,
          name: lab.title.substring(0, 20) + (lab.title.length > 20 ? '...' : ''),
          phase: 1,
          points: lab.points,
          type: lab.type,
          parent: 'phase1',
          url: `/labs/${lab.code}`
        }))
      },
      {
        id: 'phase2',
        code: 'P2',
        name: 'Phase 2: Machine Learning',
        phase: 2,
        points: 120,
        type: 'phase',
        parent: 'root',
        children: phase2Labs.map(lab => ({
          id: lab.id,
          code: lab.code,
          name: lab.title.substring(0, 20) + (lab.title.length > 20 ? '...' : ''),
          phase: 2,
          points: lab.points,
          type: lab.type,
          parent: 'phase2',
          url: `/labs/${lab.code}`
        }))
      },
      {
        id: 'phase3',
        code: 'P3',
        name: 'Phase 3: Deep Learning',
        phase: 3,
        points: 124,
        type: 'phase',
        parent: 'root',
        children: phase3Labs.map(lab => ({
          id: lab.id,
          code: lab.code,
          name: lab.title.substring(0, 20) + (lab.title.length > 20 ? '...' : ''),
          phase: 3,
          points: lab.points,
          type: lab.type,
          parent: 'phase3',
          url: `/labs/${lab.code}`
        }))
      }
    ]

    // Set up SVG dimensions
    const width = window.innerWidth - 20
    const height = window.innerHeight - 200

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .style('background-color', '#0f172a')

    svg.selectAll('*').remove()

    // Create tree layout
    const hierarchy = d3.hierarchy(root)
    const treeLayout = d3.tree<TreeNode>().size([width - 100, height - 100])
    const tree = treeLayout(hierarchy)

    // Add links
    const links = svg.selectAll('.link')
      .data(tree.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', d3.linkVertical<any, any>()
        .x(d => d.x)
        .y(d => d.y)
      )
      .attr('fill', 'none')
      .attr('stroke', (d: any) => {
        const phase = d.target.data.phase
        return phase === 1 ? '#3b82f6' : phase === 2 ? '#10b981' : '#a855f7'
      })
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.6)

    // Add nodes
    const nodes = svg.selectAll('.node')
      .data(tree.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.x},${d.y})`)

    // Node circles
    nodes.append('circle')
      .attr('r', (d: any) => {
        if (d.data.type === 'root') return 25
        if (d.data.type === 'phase') return 20
        return 12
      })
      .attr('fill', (d: any) => {
        if (d.data.type === 'root') return '#fbbf24'
        if (d.data.phase === 1) return '#3b82f6'
        if (d.data.phase === 2) return '#10b981'
        if (d.data.phase === 3) return '#a855f7'
        return '#6b7280'
      })
      .attr('stroke', 'white')
      .attr('stroke-width', 2)
      .style('cursor', (d: any) => d.data.url ? 'pointer' : 'default')
      .on('click', (event: any, d: any) => {
        if (d.data.url) {
          navigate(d.data.url)
        }
      })
      .on('mouseenter', function(event: any, d: any) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', (node: any) => {
            if (node.data.type === 'root') return 30
            if (node.data.type === 'phase') return 25
            return 16
          })
      })
      .on('mouseleave', function(event: any, d: any) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', (node: any) => {
            if (node.data.type === 'root') return 25
            if (node.data.type === 'phase') return 20
            return 12
          })
      })

    // Node labels
    nodes.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.3em')
      .attr('font-size', (d: any) => {
        if (d.data.type === 'root') return '12px'
        if (d.data.type === 'phase') return '11px'
        return '9px'
      })
      .attr('fill', 'white')
      .attr('font-weight', (d: any) => d.data.type === 'root' ? 'bold' : 'normal')
      .text((d: any) => {
        if (d.data.type === 'root') return d.data.code
        if (d.data.type === 'phase') return d.data.code
        return d.data.code
      })
      .style('pointer-events', 'none')

    // Add tooltips
    nodes.append('title')
      .text((d: any) => `${d.data.name}\n${d.data.points} points`)

  }, [navigate])

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
              Labs
            </button>
            <button
              onClick={() => navigate('/dashboard')}
              className="px-4 py-2 bg-slate-700 text-white rounded hover:bg-slate-600 transition"
            >
              Dashboard
            </button>
          </div>
        </div>
      </nav>

      {/* Title */}
      <div className="bg-slate-800 border-b border-slate-700 px-4 sm:px-6 lg:px-8 py-6">
        <h2 className="text-2xl font-bold text-white mb-2">15-Month ML/DL Program Structure</h2>
        <p className="text-gray-400">Interactive tree visualization showing all phases and labs. Click on any lab to view details.</p>
      </div>

      {/* Tree Visualization */}
      <div className="w-full overflow-auto bg-gradient-to-br from-slate-900 to-slate-800">
        <svg ref={svgRef} style={{ minHeight: 'calc(100vh - 200px)' }}></svg>
      </div>

      {/* Legend */}
      <div className="bg-slate-800 border-t border-slate-700 px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex flex-wrap gap-6">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full" style={{ backgroundColor: '#fbbf24' }}></div>
            <span className="text-gray-300 text-sm">Root Program</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full" style={{ backgroundColor: '#3b82f6' }}></div>
            <span className="text-gray-300 text-sm">Phase 1: Foundations</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full" style={{ backgroundColor: '#10b981' }}></div>
            <span className="text-gray-300 text-sm">Phase 2: ML</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full" style={{ backgroundColor: '#a855f7' }}></div>
            <span className="text-gray-300 text-sm">Phase 3: Deep Learning</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LabTree
