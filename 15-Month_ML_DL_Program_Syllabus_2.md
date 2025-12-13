# 15-Month Machine Learning & Deep Learning Program Syllabus

## Program Overview

**Total Duration:** 15 months  
**Structure:** 3 main phases  
**Format:** Each topic = 1 Lecture (theory) + 1 Practice (hands-on)  
**Target Audience:** CS students, engineering students, technical career-switchers  
**Teaching Style:** Rigorous, conceptually deep, implementation-oriented  
**Primary Framework:** PyTorch  
**Secondary Framework:** TensorFlow/Keras (comparative)  
**Assessment:** Exams/quizzes, coding assignments, projects, capstone

### Program Philosophy

- **Depth Expectation:** Students must be able to derive algorithms, implement core methods from scratch, then compare with standard libraries
- **Mathematics:** Taught in separate syllabus (prerequisite for Phase 2)
- **Machine Learning Start:** Month 9 (Phase 2)
- **Deployment/MLOps:** Optional, light exposure only

---

## Phase 1: Computational & Programming Foundations

**Duration:** Months 1-8  
**Goal:** Build strong engineering, systems, and data foundations required for ML and DL

### Month 1: Introduction & Computer Science Fundamentals

#### Week 1-2: Introduction to AI and Computing

- **Lecture 1:** What is AI, ML, DL
  - Historical context and evolution
  - Application domains (CV, NLP, Robotics, GenAI)
  - Ethical considerations in AI
  - Learning paradigms (supervised, unsupervised, reinforcement learning)
  - ML problem types (classification, regression, clustering, ranking)
- **Practice 1:**
  - Explore AI applications in different domains
  - Case study analysis
  - Ethical dilemma discussions

#### Week 3-4: Fundamentals of Computer Science

- **Lecture 2:** Programs and Algorithms
  - Algorithmic thinking
  - Computational complexity (intuitive introduction)
  - Data representation (bits, bytes, encoding)
  - Floating-point representation and numerical errors
  - Memory abstraction and addressing
- **Practice 2:**
  - Implement basic algorithms from scratch
  - Explore floating-point precision issues
  - Memory profiling exercises

---

### Month 2: Computer Architecture & Operating Systems

#### Week 1-2: Computer Architecture & OS Interfaces

- **Lecture 3:** CPU and Memory Architecture
  - CPU architecture fundamentals
  - Memory hierarchy (cache, RAM, storage)
  - Storage systems (HDD, SSD)
  - GPU architecture (threads, warps, parallelism)
  - Why GPUs matter for ML/DL
- **Practice 3:**
  - Benchmark CPU vs GPU operations
  - Memory access pattern experiments
  - Cache locality exercises

#### Week 3-4: Operating Systems Concepts

- **Lecture 4:** OS Fundamentals

  - Processes vs threads
  - System calls and kernel interaction
  - I/O abstraction

- **Practice 4:**
  - Process and thread management in Python
  - System call tracing
  - I/O performance experiments

---

### Month 3: Developer Environment & Python Fundamentals

#### Week 1: Linux Essentials & Git

- **Lecture 5:** Developer Environment Setup
  - Linux file system basics (essential commands only)
  - Package management (pip, conda)
  - Git fundamentals (init, add, commit, branches, merges)
  - GitHub workflow
  - Repository hygiene
- **Practice 5:**
  - Environment setup and configuration
  - Git workflow exercises
  - Collaborative project setup

#### Week 2-3: Python Built-in Types I & II

- **Lecture 6:** Numeric and String Types
  - Python execution model
  - Numeric types: int, float
    - Integer representation and operations
    - Floating-point precision and limitations
    - Type conversions
  - String type: str
    - String creation and manipulation
    - String methods and formatting
    - String immutability
    - Encoding and Unicode
- **Practice 6:**

  - Numeric computation exercises
  - String manipulation challenges
  - Type conversion and formatting

- **Lecture 7:** Sequence Types: Lists and Tuples
  - List type: list
    - List creation and indexing
    - List methods (append, extend, insert, remove, etc.)
    - List comprehensions
    - Mutability and references
  - Tuple type: tuple
    - Tuple creation and immutability
    - Tuple unpacking
    - When to use tuples vs lists
  - Sequence operations (slicing, concatenation, repetition)
- **Practice 7:**
  - List manipulation exercises
  - List comprehension challenges
  - Tuple unpacking and sequence operations

#### Week 4: Python Built-in Types III & Control Flow

- **Lecture 8:** Mapping and Set Types
  - Dictionary type: dict
    - Dictionary creation and access
    - Dictionary methods (keys, values, items, get, update, etc.)
    - Dictionary comprehensions
    - Hashable keys and hash tables
  - Set type: set
    - Set creation and operations
    - Set methods (add, remove, union, intersection, difference)
    - Set comprehensions
    - Use cases for sets (uniqueness, membership testing)
- **Practice 8:**

  - Dictionary manipulation exercises
  - Set operations and algorithms
  - Data structure selection challenges

- **Lecture 9:** Control Flow and Functions
  - Control flow (if/else, for, while loops)
  - Loop control (break, continue, else clauses)
  - Functions: definition, parameters, return values
  - Function arguments (positional, keyword, default, \*args, \*\*kwargs)
  - Recursion
  - Scope and namespaces (local, global, nonlocal)
- **Practice 9:**
  - Control flow exercises
  - Function implementation challenges
  - Recursive problem solving
  - Scope and namespace experiments

---

### Month 4: Advanced Python & Comprehensions

#### Week 1-2: Comprehensions and Functional Programming

- **Lecture 10:** Comprehensions and Functional Patterns
  - List comprehensions (advanced patterns)
  - Dictionary comprehensions
  - Set comprehensions
  - Generator expressions
  - Higher-order functions (map, filter, reduce)
  - Lambda functions
  - Functional programming patterns
- **Practice 10:**
  - Complex comprehension exercises
  - Functional programming challenges
  - Data transformation pipelines

#### Week 3-4: Advanced Python Features

- **Lecture 11:** Iterables, Iterators, and Generators
  - Iterables vs iterators
  - Iterator protocol (`__iter__`, `__next__`)
  - Generator functions (yield keyword)
  - Generator expressions
  - Building custom iterators
  - Generator-based data pipelines
- **Practice 11:**

  - Build iterator classes
  - Implement generator functions
  - Create generator-based data processing pipelines
  - Memory-efficient data processing

- **Lecture 12:** Decorators and Advanced Functions
  - Function decorators (syntax and implementation)
  - Decorator patterns
  - Built-in decorators (@property, @staticmethod, @classmethod)
  - Decorator factories
  - Function introspection
  - Closures and nested functions
- **Practice 12:**
  - Implement custom decorators
  - Build decorator-based utilities
  - Advanced function manipulation

---

### Month 5: Object-Oriented Programming (4 Lessons)

#### Week 1: OOP Foundations

- **Lecture 13:** Classes and Objects
  - Classes and objects fundamentals
  - Instance attributes and methods
  - `__init__` constructor
  - `self` parameter
  - Instance vs class attributes
  - Method calls and attribute access
- **Practice 13:**

  - Create simple classes
  - Implement basic OOP structures
  - Attribute and method exercises

- **Lecture 14:** Special Methods and Operator Overloading
  - Special methods (`__str__`, `__repr__`, `__len__`, `__eq__`, etc.)
  - Operator overloading (`__add__`, `__sub__`, `__mul__`, etc.)
  - Context managers (`__enter__`, `__exit__`)
  - Property decorators (`@property`, `@setter`, `@deleter`)
  - String representation methods
- **Practice 14:**
  - Implement special methods
  - Create operator-overloaded classes
  - Build context managers
  - Property-based classes

#### Week 2: Inheritance and Polymorphism

- **Lecture 15:** Inheritance and Method Resolution
  - Class inheritance
  - Method overriding
  - `super()` function
  - Method Resolution Order (MRO)
  - Multiple inheritance
  - Abstract base classes (ABC)
- **Practice 15:**

  - Build inheritance hierarchies
  - Implement method overriding
  - Work with multiple inheritance
  - Create abstract base classes

- **Lecture 16:** Composition, Design Patterns, and ML Applications
  - Composition vs inheritance
  - Design patterns in Python
    - Factory pattern
    - Singleton pattern
    - Strategy pattern
  - OOP patterns in ML systems
    - Dataset classes
    - Model classes
    - Trainer classes
    - Callback patterns
- **Practice 16:**
  - Design OOP systems with composition
  - Implement design patterns
  - Build ML-related classes (Dataset, Model, Trainer)
  - Create callback systems

---

### Month 6: Python Engineering & Data Handling

#### Week 1-2: Python Engineering Practices

- **Lecture 17:** Production-Ready Python
  - Modules and packages
  - Package structure and `__init__.py`
  - Import system and import paths
  - Virtual environments (venv, conda)
  - Dependency management (requirements.txt, environment.yml)
  - Exception handling (try/except/finally)
  - Custom exceptions
  - Debugging techniques (pdb, logging)
  - Logging best practices
  - Code organization and structure
  - PEP 8 style guide
- **Practice 17:**
  - Build a Python package
  - Set up development environments
  - Error handling and logging implementation
  - Debugging sessions
  - Code style and organization

#### Week 3-4: File Handling and Text Processing

- **Lecture 18:** Data I/O and Processing
  - File I/O (reading/writing text, binary files)
  - CSV handling (csv module, pandas)
  - JSON handling (json module)
  - Regular expressions (re module)
  - Parsing structured and semi-structured data
  - Data cleaning techniques
  - Encoding issues (UTF-8, ASCII, encoding errors)
  - File path handling (pathlib)
- **Practice 18:**
  - Build data parsers
  - Text processing pipelines
  - Data cleaning scripts
  - Regex exercises
  - File I/O challenges

---

### Month 7: Performance & Parallelism

#### Week 1-2: Parallelism and Performance

- **Lecture 19:** Concurrency in Python
  - Global Interpreter Lock (GIL)
  - Multithreading vs multiprocessing
  - CPU-bound vs I/O-bound workloads
  - Threading module (threading.Thread)
  - Multiprocessing module (Process, Pool)
  - Concurrent.futures (ThreadPoolExecutor, ProcessPoolExecutor)
  - GPU programming concepts (Python-level)
  - Introduction to CUDA concepts
  - NumPy and GPU acceleration
- **Practice 19:**
  - Benchmark sequential vs parallel code
  - Implement parallel data processing
  - Threading and multiprocessing exercises
  - GPU memory management basics
  - Performance profiling

#### Week 3-4: Data Stack (Pre-ML)

- **Lecture 20:** NumPy and Vectorization
  - NumPy arrays and operations
  - Array creation and manipulation
  - Broadcasting rules
  - Vectorization techniques
  - Array indexing and slicing (fancy indexing, boolean indexing)
  - Performance optimization
  - NumPy best practices
  - Memory layout and views
- **Practice 20:**
  - NumPy exercises
  - Vectorization vs loops comparison
  - Array manipulation challenges
  - Performance optimization exercises

---

### Month 8: Data Analysis & Databases

#### Week 1-2: Pandas and Data Analysis

- **Lecture 21:** Pandas and EDA
  - Pandas DataFrames and Series
  - Data manipulation (merge, groupby, pivot, transform)
  - Data cleaning techniques (handling missing values, duplicates)
  - Data visualization (Matplotlib, Seaborn)
  - Exploratory Data Analysis (EDA) methodology
  - Statistical summaries and analysis
  - Time series handling
- **Practice 21:**
  - Pandas data manipulation exercises
  - Build EDA pipelines
  - Visualization projects
  - **Mini-Project:** Full EDA pipeline on real dataset

#### Week 3-4: Databases and APIs

- **Lecture 22:** Data Storage and Retrieval
  - SQL fundamentals
  - Joins, aggregates, indexes
  - Database connections (SQLite, PostgreSQL basics)
  - REST APIs
  - JSON data handling
  - API design principles
  - HTTP requests (requests library)
- **Practice 22:**
  - SQL query exercises
  - Build REST API clients
  - Database integration
  - **Mini-Project:** API → database → analysis → visualization pipeline

---

## Phase 2: Machine Learning

**Duration:** Months 9-12  
**Prerequisites:** Phase 1 completed + Separate math syllabus (linear algebra, probability theory, calculus)

### Month 9: Introduction to ML & Regression

#### Week 1-2: Introduction to Machine Learning

- **Lecture 23:** ML Foundations and Learning Theory
  - Formal definition of ML
  - Learning paradigms (supervised, unsupervised, semi-supervised, reinforcement)
  - ML workflow (data → preprocessing → model → evaluation → deployment)
  - Bias-variance decomposition (mathematical formulation)
  - Bias-variance tradeoff (practical implications)
  - Learning theory basics (PAC learning, VC dimension - intuitive)
  - No-free-lunch theorem (intuitive)
  - Empirical risk vs true risk
  - Overfitting and underfitting (formal definitions)
- **Practice 23:**
  - Set up ML development environment
  - Explore ML datasets
  - Implement basic data preprocessing pipeline
  - Visualize bias-variance tradeoff

#### Week 3-4: Supervised Learning I — Regression

- **Lecture 24:** Linear and Polynomial Regression
  - Linear regression (mathematical formulation)
  - Least squares solution (derivation)
  - Normal equation derivation
  - Polynomial regression
  - Bias–variance tradeoff (detailed analysis)
  - Overfitting and underfitting (visualization and analysis)
  - Regularization (Ridge, Lasso) — mathematical derivation
  - Elastic Net regularization
  - Regression metrics (MSE, MAE, RMSE, R², adjusted R²)
  - Feature scaling and normalization
- **Practice 24:**
  - Implement linear regression from scratch
  - Implement Ridge and Lasso from scratch
  - Compare with scikit-learn
  - Experiment with bias-variance tradeoff
  - Regularization parameter tuning

---

### Month 10: Model Evaluation & Classification

#### Week 1-2: Model Evaluation & Validation

- **Lecture 25:** Evaluation Strategies
  - Train/validation/test split
  - Cross-validation (k-fold, stratified, leave-one-out, time-series CV)
  - Error analysis techniques
  - Learning curves (training vs validation)
  - Hyperparameter tuning strategies (grid search, random search, Bayesian optimization)
  - Model selection and comparison
  - Statistical significance testing
- **Practice 25:**
  - Implement cross-validation from scratch
  - Build hyperparameter tuning pipeline
  - Error analysis exercises
  - Learning curve visualization

#### Week 3-4: Supervised Learning II — Classification

- **Lecture 26:** Classification Algorithms
  - Logistic regression (mathematical derivation)
  - Maximum likelihood estimation
  - k-Nearest Neighbors (k-NN)
  - Decision boundaries
  - Classification metrics:
    - Accuracy, precision, recall, F1
    - Confusion matrix
    - ROC curve and AUC
    - Precision-recall curve
    - Multi-class metrics
  - Generative vs discriminative learning (introduction)
- **Practice 26:**
  - Implement logistic regression from scratch
  - Implement k-NN from scratch
  - Build metric calculation functions
  - Compare with scikit-learn
  - Visualize decision boundaries

---

### Month 11: Optimization & Advanced Supervised Learning

#### Week 1-2: Optimization Techniques

- **Lecture 27:** Gradient-Based Optimization
  - Gradient Descent (derivation)
  - Stochastic Gradient Descent (SGD)
  - Mini-batch gradient descent
  - Momentum (Nesterov momentum)
  - Adaptive methods: Adam, RMSProp, AdaGrad (derivations)
  - Learning rate scheduling (step decay, exponential decay, cosine annealing)
  - Second-order methods (Newton's method, quasi-Newton methods - overview)
  - Optimization for non-convex problems
- **Practice 27:**
  - Implement gradient descent from scratch
  - Implement SGD, Adam, RMSProp
  - Visualize optimization paths
  - Compare optimization algorithms
  - Learning rate scheduling experiments

#### Week 3-4: Support Vector Machines

- **Lecture 28:** SVM Theory and Practice
  - SVM for classification (mathematical formulation)
  - Hard margin vs soft margin
  - SVM for regression (SVR)
  - Kernel trick (mathematical intuition and derivation)
  - Common kernels (linear, polynomial, RBF, sigmoid)
  - Dual formulation (Lagrangian duality)
  - Kernel methods overview
- **Practice 28:**
  - Implement SVM from scratch (simplified)
  - Kernel implementation
  - Compare with scikit-learn
  - Visualize decision boundaries with different kernels

---

### Month 12: Ensemble Methods & Unsupervised Learning

#### Week 1-2: Decision Trees and Ensemble Methods

- **Lecture 29:** Tree-Based Methods
  - Decision trees (ID3, CART, C4.5)
  - Splitting criteria (entropy, Gini impurity, information gain)
  - Pruning (pre-pruning, post-pruning)
  - Random Forests (bootstrap aggregation, feature bagging)
  - Gradient Boosting (XGBoost, LightGBM concepts)
  - AdaBoost
  - Model comparison and selection
  - Feature importance
- **Practice 29:**
  - Implement decision tree from scratch
  - Implement random forest
  - Implement gradient boosting (simplified)
  - Compare ensemble methods
  - Feature importance analysis

#### Week 3-4: Unsupervised Learning

- **Lecture 30:** Clustering and Dimensionality Reduction
  - K-Means clustering (algorithm derivation, initialization methods)
  - Hierarchical clustering (agglomerative, divisive)
  - DBSCAN (density-based clustering)
  - Principal Component Analysis (PCA) — mathematical derivation
  - Singular Value Decomposition (SVD) for dimensionality reduction
  - Gaussian Mixture Models (GMM)
  - Expectation-Maximization (EM) algorithm (derivation)
  - t-SNE and UMAP (overview)
- **Practice 30:**
  - Implement K-Means from scratch
  - Implement PCA from scratch
  - Implement GMM with EM
  - Compare with scikit-learn
  - **Machine Learning Mini-Project:** End-to-end ML pipeline with presentation and evaluation

---

## Phase 3: Deep Learning & Advanced AI

**Duration:** Months 13-15

### Month 13: Deep Learning Fundamentals & Computer Vision

#### Week 1-2: Deep Learning Fundamentals

- **Lecture 31:** Neural Networks Foundations
  - Perceptron and multi-layer perceptron
  - Forward propagation (mathematical derivation)
  - Backward propagation (backpropagation derivation)
  - Loss functions (MSE, cross-entropy, hinge loss)
  - Activation functions (sigmoid, tanh, ReLU, Leaky ReLU, ELU, Swish)
  - Weight initialization (Xavier/Glorot, He initialization)
  - Optimization for deep learning
  - Vanishing/exploding gradients
  - Gradient clipping
- **Practice 31:**
  - Implement neural network from scratch (NumPy)
  - Implement backpropagation manually
  - Compare with PyTorch
  - Visualize gradients and activations
  - Experiment with different initialization methods

#### Week 3-4: Deep Learning for Computer Vision

- **Lecture 32:** Convolutional Neural Networks
  - CNN fundamentals (convolution, pooling, stride, padding)
  - CNN architectures: LeNet, AlexNet, VGG, GoogLeNet, ResNet
  - Normalization (BatchNorm, LayerNorm, GroupNorm, InstanceNorm)
  - Regularization techniques (Dropout, DropConnect, Data Augmentation)
  - Image classification pipeline
  - Transfer learning strategies
- **Practice 32:**
  - Implement CNN from scratch (PyTorch)
  - Train on image classification dataset
  - Implement data augmentation
  - Compare architectures
  - Transfer learning experiments

---

### Month 14: Advanced Computer Vision & Generative AI

#### Week 1-2: Advanced Computer Vision

- **Lecture 33:** Advanced CV Techniques
  - Transfer learning (fine-tuning strategies)
  - Image retrieval
  - Semantic segmentation (UNet, FCN architectures)
  - Instance segmentation (Mask R-CNN)
  - Object detection (SSD, YOLO, R-CNN, Faster R-CNN, DETR)
  - Feature extraction and visualization
  - Attention mechanisms in vision
- **Practice 33:**

  - Implement transfer learning pipeline
  - Build image retrieval system
  - Implement UNet for segmentation
  - Object detection with pre-trained models
  - Feature visualization

- **Lecture 34:** Self-Supervised Learning
  - Self-supervised learning principles
  - Pretext tasks (rotation, jigsaw, colorization)
  - Contrastive learning (SimCLR, MoCo concepts)
  - Vision Transformers (ViT)
  - Self-supervised pre-training for downstream tasks
- **Practice 34:**
  - Implement self-supervised learning tasks
  - Contrastive learning experiments
  - Fine-tune self-supervised models

#### Week 3-4: Generative AI (Vision)

- **Lecture 35:** Generative Models I
  - Variational Autoencoders (VAE) — mathematical formulation
  - ELBO derivation
  - Generative Adversarial Networks (GANs)
  - GAN training dynamics
  - Autoregressive models (PixelCNN, PixelRNN)
- **Practice 35:**

  - Implement VAE from scratch
  - Implement GAN from scratch
  - Train generative models
  - Evaluate generative models

- **Lecture 36:** Generative Models II
  - Diffusion models (DDPM, DDIM)
  - Score-based generative models
  - Stable Diffusion architecture
  - Normalizing flows (overview)
  - Energy-based models (overview)
  - Evaluation metrics for generative models (FID, IS, etc.)
- **Practice 36:**
  - Train diffusion model (simplified)
  - Generate images with Stable Diffusion
  - Evaluate generative models
  - Compare different generative approaches

---

### Month 15: NLP, Multi-Modal AI & Capstone

#### Week 1-2: NLP and Multi-Modal AI

- **Lecture 37:** Natural Language Processing
  - Word embeddings (Word2Vec, GloVe, FastText)
  - Text preprocessing and tokenization
  - RNNs and LSTM networks
  - GRU networks
  - Transformers architecture (attention mechanism, self-attention, multi-head attention)
  - Positional encoding
  - BERT, GPT architectures
  - Fine-tuning strategies
- **Practice 37:**

  - Implement word embeddings
  - Build LSTM/GRU for text classification
  - Implement transformer components from scratch
  - Fine-tune BERT/GPT
  - Text generation experiments

- **Lecture 38:** Multi-Modal AI
  - CLIP (Contrastive Language-Image Pre-training)
  - Vision-Language models
  - Image captioning (encoder-decoder, attention mechanisms)
  - Visual Question Answering (VQA)
  - Multi-modal fusion strategies
  - Large-scale pre-training
- **Practice 38:**
  - Build image captioning system
  - CLIP-based applications
  - VQA implementation
  - Multi-modal experiments

#### Week 3-4: Capstone Project

- **Lecture 39:** Project Planning and Best Practices
  - Project scoping
  - Literature review
  - System design
  - Evaluation methodologies
  - Presentation skills
  - Reproducibility and best practices
- **Practice 39:** Capstone Project
  - Individual or team-based (2-4 students)
  - Project planning and proposal
  - Implementation
  - Evaluation and analysis
  - Final presentation and demonstration

---

## Expected Graduate Profile

By the end of the 15-month program, students should:

### Technical Skills

- ✅ Strong systems and programming foundations
- ✅ Deep understanding of ML and DL theory
- ✅ Ability to derive and implement algorithms from scratch
- ✅ Proficiency in PyTorch with comparative knowledge of TensorFlow
- ✅ Experience building end-to-end ML/DL systems
- ✅ Solid software engineering practices

### Theoretical Understanding

- ✅ Mathematical foundations (via separate syllabus)
- ✅ Algorithm derivations and proofs
- ✅ Understanding of bias-variance tradeoffs
- ✅ Knowledge of optimization theory
- ✅ Understanding of neural network theory

### Practical Capabilities

- ✅ Can implement core ML/DL algorithms from scratch
- ✅ Can compare custom implementations with standard libraries
- ✅ Can build complete ML pipelines
- ✅ Can evaluate and debug models
- ✅ Can work with real-world datasets

### Career Readiness

- ✅ Ready for ML Engineer roles
- ✅ Ready for Applied AI Engineer roles
- ✅ Prepared for advanced research
- ✅ Strong portfolio of projects
- ✅ Capstone project demonstrating end-to-end capabilities

---

## Assessment Structure

### Continuous Assessment

- **Quizzes:** Theory and conceptual understanding (weekly/bi-weekly)
- **Coding Assignments:** Implementation from scratch (after each major topic)
- **Mini-Projects:** End-to-end pipelines (integrated throughout phases)
- **Peer Reviews:** Code review and collaboration exercises

---

## Phase Examinations

### Phase 1 Final Examination

**Duration:** 2 weeks  
**Weight:** 30% of Phase 1 grade  
**Format:** Combined Implementation Project + Technical Interview

#### Component 1: Implementation Project (70% of exam grade)

**Duration:** 10 days  
**Deliverables:**

- Build a complete Python application from scratch
- Implement a data processing pipeline (CSV/JSON → cleaning → analysis → visualization)
- Demonstrate proficiency in:
  - OOP design patterns
  - Error handling and logging
  - Code organization and documentation
  - Git version control
  - Testing (unit tests)
- **Requirements:**
  - Minimum 500 lines of well-documented code
  - Use of at least 3 different Python modules
  - Proper package structure
  - GitHub repository with commit history
  - README with setup instructions
  - Code review by peers

#### Component 2: Technical Interview (30% of exam grade)

**Duration:** 45-60 minutes  
**Format:** One-on-one with instructor/TA  
**Topics Covered:**

- Python fundamentals (types, control flow, functions)
- OOP concepts (inheritance, polymorphism, design patterns)
- Data structures and algorithms (basic complexity analysis)
- System design (file I/O, modules, packages)
- Code review of submitted project
- Debugging scenarios
- **Evaluation Criteria:**
  - Conceptual understanding
  - Problem-solving approach
  - Communication skills
  - Code quality awareness

---

### Phase 2 Final Examination

**Duration:** 2 weeks  
**Weight:** 35% of Phase 2 grade  
**Format:** Combined Implementation Project + Comprehensive Quiz

#### Component 1: ML Implementation Project (60% of exam grade)

**Duration:** 12 days  
**Deliverables:**

- End-to-end machine learning pipeline
- **Requirements:**
  - Implement at least 3 algorithms from scratch (e.g., linear regression, logistic regression, decision tree)
  - Compare implementations with scikit-learn
  - Apply to real-world dataset
  - Complete EDA and preprocessing
  - Model evaluation and hyperparameter tuning
  - Error analysis and interpretation
- **Deliverables:**
  - Jupyter notebook with complete analysis
  - Well-documented code with mathematical derivations
  - Written report (5-8 pages) explaining:
    - Algorithm derivations
    - Implementation decisions
    - Results and analysis
    - Comparison with library implementations
  - Presentation (10-15 minutes)
- **Evaluation Criteria:**
  - Correctness of implementations
  - Mathematical rigor
  - Code quality and documentation
  - Analysis depth
  - Presentation clarity

#### Component 2: Comprehensive Quiz (40% of exam grade)

**Duration:** 2 hours (closed book, with formula sheet)  
**Format:** Written exam  
**Topics Covered:**

- Mathematical derivations (linear regression, logistic regression, SVM, PCA)
- Algorithm understanding (gradient descent, backpropagation, EM algorithm)
- Bias-variance tradeoff
- Model evaluation metrics
- Optimization theory
- Learning theory basics
- **Question Types:**
  - Derive algorithm steps
  - Explain mathematical concepts
  - Analyze algorithm complexity
  - Compare different approaches
  - Solve optimization problems
- **Evaluation Criteria:**
  - Mathematical accuracy
  - Conceptual understanding
  - Problem-solving approach
  - Clarity of explanation

---

### Phase 3 Final Examination

**Duration:** 4 weeks (Capstone) + 1 week (Interview)  
**Weight:** 40% of Phase 3 grade  
**Format:** Capstone Project + Technical Deep-Dive Interview

#### Component 1: Capstone Project (75% of exam grade)

**Duration:** 4 weeks  
**Format:** Individual or team-based (2-4 students)  
**Requirements:**

- Complete deep learning project from conception to deployment-ready solution
- **Project Scope:**
  - Real-world problem definition
  - Literature review (5-10 papers)
  - System design and architecture
  - Implementation using PyTorch
  - Comprehensive evaluation
  - Comparison with baseline methods
- **Deliverables:**
  - Project proposal (2-3 pages)
  - Progress report (mid-point, 3-5 pages)
  - Final codebase (well-documented, reproducible)
  - Final report (10-15 pages) including:
    - Problem statement and motivation
    - Related work
    - Methodology
    - Experiments and results
    - Analysis and discussion
    - Future work
  - Final presentation (20-30 minutes)
  - Demo/video (if applicable)
  - GitHub repository with full documentation
- **Evaluation Criteria:**
  - Technical depth and innovation
  - Implementation quality
  - Experimental rigor
  - Analysis and interpretation
  - Presentation and communication
  - Reproducibility

#### Component 2: Technical Deep-Dive Interview (25% of exam grade)

**Duration:** 60-90 minutes  
**Format:** Panel interview with instructors/TAs  
**Topics Covered:**

- Deep dive into capstone project:
  - Architecture decisions
  - Implementation challenges
  - Experimental design
  - Results interpretation
- Deep learning theory:
  - Backpropagation derivation
  - Optimization for deep learning
  - Regularization techniques
  - Architecture design principles
- Advanced topics:
  - Transformers and attention mechanisms
  - Generative models (VAE, GAN, Diffusion)
  - Transfer learning and fine-tuning
  - Multi-modal learning
- Code review of capstone implementation
- Debugging and optimization scenarios
- **Evaluation Criteria:**
  - Deep technical understanding
  - Ability to explain complex concepts
  - Problem-solving under pressure
  - Critical thinking
  - Communication skills

---

## Overall Assessment Summary

### Grade Distribution

**Phase 1:**

- Continuous Assessment: 70%
- Phase 1 Final Exam: 30%

**Phase 2:**

- Continuous Assessment: 65%
- Phase 2 Final Exam: 35%

**Phase 3:**

- Continuous Assessment: 60%
- Phase 3 Final Exam (Capstone + Interview): 40%

### Evaluation Criteria (Applied Across All Assessments)

- **Theory:** Ability to derive and explain algorithms mathematically
- **Implementation:** Code quality, correctness, efficiency, best practices
- **Analysis:** Critical thinking, error analysis, interpretation of results
- **Communication:** Presentation skills, documentation quality, clarity of explanation
- **Innovation:** Creative problem-solving, application of concepts to new problems
- **Collaboration:** Teamwork, code review, peer learning (where applicable)

---

## Resources and Tools

### Primary Tools

- **Languages:** Python 3.x
- **ML Framework:** PyTorch (primary), TensorFlow/Keras (secondary)
- **Data Science:** NumPy, Pandas, Matplotlib, Seaborn
- **ML Libraries:** scikit-learn (for comparison)
- **Version Control:** Git, GitHub
- **Environment:** Linux, Conda/Venv

### Recommended Learning Resources

- Course materials and lecture notes
- Research papers (selected)
- Open-source implementations
- Real-world datasets

---

## Notes

- **Mathematics:** All mathematical prerequisites are covered in a separate syllabus running in parallel or as prerequisite
- **Flexibility:** Some topics may be adjusted based on cohort needs
- **Depth:** Emphasis on deep understanding over breadth
- **Projects:** All projects use real-world datasets
- **Deployment:** MLOps and deployment are optional/light exposure only

---

_This syllabus is designed to produce graduates who are both theoretically grounded and practically capable, ready to contribute to the ML/DL field at a professional level._
