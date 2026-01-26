export interface Task {
  title: string
  description: string
  deliverables: string[]
}

export interface Lab {
  id: string
  code: string
  phase: 1 | 2 | 3
  title: string
  description: string
  points: number
  type: 'mandatory' | 'optional'
  selfStudyMaterials: string[]
  requiredKnowledge: string[]
  tasks: Task[]
  notes?: string
}

export const labs: Lab[] = [
  {
    id: '1-1',
    code: '1.1',
    phase: 1,
    title: 'Python Fundamentals Deep Dive',
    description: 'Implement data structures and algorithms from scratch, understand Python internals',
    points: 12,
    type: 'mandatory',
    selfStudyMaterials: ['Python tutorials', 'CS106A/B materials', 'Algorithm visualization'],
    requiredKnowledge: ['Python basics', 'Data structures', 'Algorithmic thinking', 'Complexity concepts', 'Testing', 'Git'],
    tasks: [
      {
        title: 'Data Structures Implementation',
        description: 'Implement Stack, Queue, Linked List, Binary Tree, Hash Table from scratch.',
        deliverables: ['GitHub repo', 'Performance analysis report']
      },
      {
        title: 'Algorithm Implementation',
        description: 'Implement sorting and search algorithms.',
        deliverables: ['Jupyter notebook', 'Complexity analysis', 'Visualizations']
      }
    ]
  },
  {
    id: '1-2',
    code: '1.2',
    phase: 1,
    title: 'System Programming & Performance',
    description: 'Master memory management, concurrency, and performance optimization',
    points: 12,
    type: 'mandatory',
    selfStudyMaterials: ['OS concepts', 'Python internals', 'High Performance Python'],
    requiredKnowledge: ['Python fundamentals', 'Processes and threads', 'Memory management', 'Profiling tools'],
    tasks: [
      {
        title: 'Memory Management Project',
        description: 'Build memory profiler tool and analyze memory usage.',
        deliverables: ['Memory profiling tool', 'Analysis report']
      },
      {
        title: 'Concurrency & Parallelism',
        description: 'Implement parallel data processing system.',
        deliverables: ['Parallel framework', 'Benchmark results']
      }
    ]
  },
  {
    id: '1-3',
    code: '1.3',
    phase: 1,
    title: 'Data Structures & Algorithms Project',
    description: 'Build complete data processing system using custom data structures',
    points: 18,
    type: 'mandatory',
    selfStudyMaterials: ['CLRS algorithms', 'LeetCode patterns'],
    requiredKnowledge: ['Data structures', 'Algorithms', 'Complexity analysis', 'File I/O'],
    tasks: [
      {
        title: 'Build Data Processing System',
        description: 'Design system that processes large datasets with custom data structures.',
        deliverables: ['Complete system', 'Benchmarks', 'Design document', 'Presentation']
      }
    ]
  },
  {
    id: '2-1',
    code: '2.1',
    phase: 2,
    title: 'Linear Models from Scratch',
    description: 'Implement regression models: OLS, Ridge, Lasso, Elastic Net',
    points: 12,
    type: 'mandatory',
    selfStudyMaterials: ['Statistical learning theory', 'Optimization textbooks', 'Elements of Statistical Learning'],
    requiredKnowledge: ['Linear algebra', 'Calculus', 'Statistics', 'NumPy', 'Optimization'],
    tasks: [
      {
        title: 'Linear Regression',
        description: 'Implement OLS, Ridge, Lasso, Elastic Net from scratch.',
        deliverables: ['Implementation', 'Mathematical derivations']
      },
      {
        title: 'Regularized Regression',
        description: 'Implement polynomial regression with cross-validation.',
        deliverables: ['Regression library', 'Analysis report']
      }
    ]
  },
  {
    id: '2-2',
    code: '2.2',
    phase: 2,
    title: 'Classification Algorithms',
    description: 'Implement classification: Logistic Regression, k-NN, Naive Bayes',
    points: 12,
    type: 'mandatory',
    selfStudyMaterials: ['Classification theory', 'Probability theory'],
    requiredKnowledge: ['Regression concepts', 'Probability', 'Distance metrics'],
    tasks: [
      {
        title: 'Logistic Regression',
        description: 'Implement with gradient descent and Newton\'s method.',
        deliverables: ['Implementation', 'Comparison']
      },
      {
        title: 'k-NN & Naive Bayes',
        description: 'Implement classifiers with optimization.',
        deliverables: ['Classification library']
      }
    ]
  },
  {
    id: '2-8',
    code: '2.8',
    phase: 2,
    title: 'Generative Models — VAE & GAN',
    description: 'Large project: Implement Variational Autoencoder and GANs',
    points: 20,
    type: 'mandatory',
    selfStudyMaterials: ['Generative models papers', 'Variational inference', 'GAN training'],
    requiredKnowledge: ['Deep learning', 'Probability', 'Autoencoders', 'PyTorch/TensorFlow'],
    tasks: [
      {
        title: 'VAE Implementation',
        description: 'Implement VAE from scratch, derive ELBO, train on images.',
        deliverables: ['VAE implementation', 'Generated samples']
      },
      {
        title: 'GAN Implementation',
        description: 'Implement GAN with different architectures.',
        deliverables: ['GAN implementation', 'Training analysis']
      }
    ]
  },
  {
    id: '3-1',
    code: '3.1',
    phase: 3,
    title: 'Neural Networks from Scratch',
    description: 'Implement fully connected networks with backpropagation in NumPy',
    points: 12,
    type: 'mandatory',
    selfStudyMaterials: ['Deep learning book', 'Backpropagation', 'Initialization strategies'],
    requiredKnowledge: ['Linear algebra', 'Calculus', 'NumPy', 'Optimization'],
    tasks: [
      {
        title: 'Network Implementation',
        description: 'Implement forward/backward propagation, activations, loss functions.',
        deliverables: ['Neural network library']
      },
      {
        title: 'Training System',
        description: 'Implement optimizers, initialization, regularization.',
        deliverables: ['Training system', 'Results']
      }
    ]
  },
  {
    id: '3-4',
    code: '3.4',
    phase: 3,
    title: 'Diffusion Models',
    description: 'Large project: Implement diffusion model for image generation',
    points: 20,
    type: 'mandatory',
    selfStudyMaterials: ['DDPM/DDIM papers', 'Score-based models', 'Diffusion variants'],
    requiredKnowledge: ['Deep learning', 'Generative models', 'Probability', 'U-Net architecture'],
    tasks: [
      {
        title: 'Implement Diffusion Model',
        description: 'Understand diffusion, implement forward/reverse process, U-Net, train.',
        deliverables: ['Model', 'Trained weights', 'Samples', 'Mathematical document']
      }
    ]
  }
]

export function getLabsByPhase(phase: 1 | 2 | 3): Lab[] {
  return labs.filter(lab => lab.phase === phase)
}

export function getLabByCode(code: string): Lab | undefined {
  return labs.find(lab => lab.code === code)
}

export function getLabById(id: string): Lab | undefined {
  return labs.find(lab => lab.id === id)
}

export function getMandatoryLabsPoints(phase: 1 | 2 | 3): number {
  return getLabsByPhase(phase)
    .filter(lab => lab.type === 'mandatory')
    .reduce((sum, lab) => sum + lab.points, 0)
}

export function getOptionalLabsPoints(phase: 1 | 2 | 3): number {
  return getLabsByPhase(phase)
    .filter(lab => lab.type === 'optional')
    .reduce((sum, lab) => sum + lab.points, 0)
}
