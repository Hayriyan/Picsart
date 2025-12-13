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

### Month 3: Linux & Developer Environment

#### Week 1-2: Linux Fundamentals
- **Lecture 5:** Linux File System and Environment
  - Linux file system structure
  - Permissions and ownership
  - Processes and environment variables
  - Package management (apt, yum, pip, conda)
  - System configuration
- **Practice 5:**
  - Linux command-line exercises
  - Environment setup and configuration
  - Package management workflows

#### Week 3-4: Bash and Shell Scripting
- **Lecture 6:** Shell Scripting Essentials
  - Essential Bash commands
  - Pipes and redirection
  - grep, find, awk, sed, xargs
  - Control flow in shell scripts
  - Scripting best practices
- **Practice 6:**
  - Write automation scripts
  - Data processing pipelines in bash
  - System administration tasks

---

### Month 4: Version Control & Python Fundamentals

#### Week 1-2: Git and Version Control
- **Lecture 7:** Git Fundamentals
  - Version control concepts
  - Git basics (init, add, commit)
  - Branches and merges
  - GitHub workflow
  - Repository hygiene and best practices
  - Collaboration patterns
- **Practice 7:**
  - Git workflow exercises
  - Branching strategies
  - Merge conflict resolution
  - Collaborative project setup

#### Week 3-4: Python Programming Fundamentals
- **Lecture 8:** Python Core Concepts
  - Python execution model
  - Built-in types (int, float, str, list, dict, tuple, set)
  - Control flow (if/else, loops)
  - Comprehensions (list, dict, set)
  - Functions and recursion
  - Scope and namespaces
- **Practice 8:**
  - Python fundamentals exercises
  - Algorithm implementation in Python
  - Recursive problem solving

---

### Month 5: Advanced Python & OOP

#### Week 1-2: Advanced Python Concepts
- **Lecture 9:** Python Advanced Features
  - Iterables, iterators, generators
  - Generator expressions
  - Higher-order functions
  - Lambda functions
  - Functional programming patterns
  - Decorators
- **Practice 9:**
  - Build iterator classes
  - Generator-based data pipelines
  - Functional programming exercises
  - Decorator implementation

#### Week 3-4: Object-Oriented Programming
- **Lecture 10:** OOP in Python
  - Classes and objects
  - `__init__` and instance methods
  - Class vs instance attributes
  - Composition vs inheritance
  - Special methods (`__str__`, `__repr__`, `__len__`, etc.)
  - OOP patterns in ML systems
- **Practice 10:**
  - Design OOP systems
  - Implement ML-related classes (Dataset, Model, Trainer)
  - Composition vs inheritance exercises

---

### Month 6: Python Engineering & Data Handling

#### Week 1-2: Python Engineering Practices
- **Lecture 11:** Production-Ready Python
  - Modules and packages
  - Virtual environments (venv, conda)
  - Exception handling
  - Debugging techniques
  - Logging best practices
  - Code organization and structure
- **Practice 11:**
  - Build a Python package
  - Set up development environments
  - Error handling and logging implementation
  - Debugging sessions

#### Week 3-4: File Handling and Text Processing
- **Lecture 12:** Data I/O and Processing
  - File I/O (CSV, JSON, text files)
  - Regular expressions
  - Parsing structured and semi-structured data
  - Data cleaning techniques
  - Encoding issues
- **Practice 12:**
  - Build data parsers
  - Text processing pipelines
  - Data cleaning scripts
  - Regex exercises

---

### Month 7: Performance & Parallelism

#### Week 1-2: Parallelism and Performance
- **Lecture 13:** Concurrency in Python
  - Global Interpreter Lock (GIL)
  - Multithreading vs multiprocessing
  - CPU-bound vs I/O-bound workloads
  - Threading module
  - Multiprocessing module
  - GPU programming concepts (Python-level)
  - Introduction to CUDA concepts
- **Practice 13:**
  - Benchmark sequential vs parallel code
  - Implement parallel data processing
  - Threading and multiprocessing exercises
  - GPU memory management basics

#### Week 3-4: Data Stack (Pre-ML)
- **Lecture 14:** NumPy and Vectorization
  - NumPy arrays and operations
  - Broadcasting rules
  - Vectorization techniques
  - Array indexing and slicing
  - Performance optimization
- **Practice 14:**
  - NumPy exercises
  - Vectorization vs loops comparison
  - Array manipulation challenges

---

### Month 8: Data Analysis & Databases

#### Week 1-2: Pandas and Data Analysis
- **Lecture 15:** Pandas and EDA
  - Pandas DataFrames
  - Data manipulation (merge, groupby, pivot)
  - Data cleaning techniques
  - Data visualization (Matplotlib, Seaborn)
  - Exploratory Data Analysis (EDA) methodology
- **Practice 15:**
  - Pandas data manipulation exercises
  - Build EDA pipelines
  - Visualization projects
  - **Mini-Project:** Full EDA pipeline on real dataset

#### Week 3-4: Databases and APIs
- **Lecture 16:** Data Storage and Retrieval
  - SQL fundamentals
  - Joins, aggregates, indexes
  - REST APIs
  - JSON data handling
  - API design principles
- **Practice 16:**
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
- **Lecture 17:** ML Foundations
  - Formal definition of ML
  - Learning paradigms (supervised, unsupervised, semi-supervised, reinforcement)
  - ML workflow (data → preprocessing → model → evaluation → deployment)
  - Bias-variance decomposition
  - No-free-lunch theorem (intuitive)
- **Practice 17:**
  - Set up ML development environment
  - Explore ML datasets
  - Implement basic data preprocessing pipeline

#### Week 3-4: Supervised Learning I — Regression
- **Lecture 18:** Linear and Polynomial Regression
  - Linear regression (mathematical formulation)
  - Least squares solution (derivation)
  - Polynomial regression
  - Bias–variance tradeoff
  - Overfitting and underfitting
  - Regularization (Ridge, Lasso) — mathematical derivation
  - Regression metrics (MSE, MAE, RMSE, R²)
- **Practice 18:**
  - Implement linear regression from scratch
  - Implement Ridge and Lasso from scratch
  - Compare with scikit-learn
  - Experiment with bias-variance tradeoff

---

### Month 10: Model Evaluation & Classification

#### Week 1-2: Model Evaluation & Validation
- **Lecture 19:** Evaluation Strategies
  - Train/validation/test split
  - Cross-validation (k-fold, stratified, leave-one-out)
  - Error analysis techniques
  - Learning curves
  - Hyperparameter tuning strategies
- **Practice 19:**
  - Implement cross-validation from scratch
  - Build hyperparameter tuning pipeline
  - Error analysis exercises

#### Week 3-4: Supervised Learning II — Classification
- **Lecture 20:** Classification Algorithms
  - Logistic regression (mathematical derivation)
  - k-Nearest Neighbors (k-NN)
  - Decision boundaries
  - Classification metrics:
    - Accuracy, precision, recall, F1
    - Confusion matrix
    - ROC curve and AUC
    - Precision-recall curve
- **Practice 20:**
  - Implement logistic regression from scratch
  - Implement k-NN from scratch
  - Build metric calculation functions
  - Compare with scikit-learn

---

### Month 11: Optimization & Advanced Supervised Learning

#### Week 1-2: Optimization Techniques
- **Lecture 21:** Gradient-Based Optimization
  - Gradient Descent (derivation)
  - Stochastic Gradient Descent (SGD)
  - Mini-batch gradient descent
  - Momentum
  - Adaptive methods: Adam, RMSProp (derivations)
  - Learning rate scheduling
- **Practice 21:**
  - Implement gradient descent from scratch
  - Implement SGD, Adam, RMSProp
  - Visualize optimization paths
  - Compare optimization algorithms

#### Week 3-4: Support Vector Machines
- **Lecture 22:** SVM Theory and Practice
  - SVM for classification (mathematical formulation)
  - Hard margin vs soft margin
  - SVM for regression (SVR)
  - Kernel trick (mathematical intuition)
  - Common kernels (linear, polynomial, RBF)
  - Dual formulation
- **Practice 22:**
  - Implement SVM from scratch (simplified)
  - Kernel implementation
  - Compare with scikit-learn
  - Visualize decision boundaries with different kernels

---

### Month 12: Ensemble Methods & Unsupervised Learning

#### Week 1-2: Decision Trees and Ensemble Methods
- **Lecture 23:** Tree-Based Methods
  - Decision trees (ID3, CART)
  - Splitting criteria (entropy, Gini impurity)
  - Pruning
  - Random Forests
  - Gradient Boosting (XGBoost concepts)
  - Model comparison and selection
- **Practice 23:**
  - Implement decision tree from scratch
  - Implement random forest
  - Implement gradient boosting (simplified)
  - Compare ensemble methods

#### Week 3-4: Unsupervised Learning
- **Lecture 24:** Clustering and Dimensionality Reduction
  - K-Means clustering (algorithm derivation)
  - Hierarchical clustering
  - DBSCAN
  - Principal Component Analysis (PCA) — mathematical derivation
  - Gaussian Mixture Models (GMM)
  - Expectation-Maximization (EM) algorithm
- **Practice 24:**
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
- **Lecture 25:** Neural Networks Foundations
  - Perceptron and multi-layer perceptron
  - Forward propagation (mathematical derivation)
  - Backward propagation (backpropagation derivation)
  - Loss functions (MSE, cross-entropy)
  - Activation functions (sigmoid, tanh, ReLU, variants)
  - Optimization for deep learning
  - Vanishing/exploding gradients
- **Practice 25:**
  - Implement neural network from scratch (NumPy)
  - Implement backpropagation manually
  - Compare with PyTorch
  - Visualize gradients and activations

#### Week 3-4: Deep Learning for Computer Vision
- **Lecture 26:** Convolutional Neural Networks
  - CNN fundamentals (convolution, pooling)
  - CNN architectures: LeNet, VGG, ResNet
  - Normalization (BatchNorm, LayerNorm)
  - Regularization techniques (Dropout, Data Augmentation)
  - Image classification pipeline
- **Practice 26:**
  - Implement CNN from scratch (PyTorch)
  - Train on image classification dataset
  - Implement data augmentation
  - Compare architectures

---

### Month 14: Advanced Computer Vision & Generative AI

#### Week 1-2: Advanced Computer Vision
- **Lecture 27:** Advanced CV Techniques
  - Transfer learning
  - Image retrieval
  - Semantic segmentation (UNet architecture)
  - Object detection (SSD, YOLO, R-CNN concepts)
  - Feature extraction and visualization
- **Practice 27:**
  - Implement transfer learning pipeline
  - Build image retrieval system
  - Implement UNet for segmentation
  - Object detection with pre-trained models

#### Week 3-4: Generative AI (Vision)
- **Lecture 28:** Generative Models
  - Variational Autoencoders (VAE) — mathematical formulation
  - Diffusion models (DDPM, DDIM)
  - Stable Diffusion architecture
  - Generative applications
  - Evaluation metrics for generative models
- **Practice 28:**
  - Implement VAE from scratch
  - Train diffusion model (simplified)
  - Generate images with Stable Diffusion
  - Evaluate generative models

---

### Month 15: NLP, Multi-Modal AI & Capstone

#### Week 1-2: NLP and Multi-Modal AI
- **Lecture 29:** Natural Language Processing
  - Word embeddings (Word2Vec, GloVe)
  - Text preprocessing
  - LSTM networks
  - Transformers architecture (attention mechanism, self-attention)
  - BERT, GPT architectures
  - CLIP (Contrastive Language-Image Pre-training)
  - Image captioning
  - Visual Question Answering (VQA)
- **Practice 29:**
  - Implement word embeddings
  - Build LSTM for text classification
  - Implement transformer components
  - Fine-tune BERT/GPT
  - Build image captioning system
  - CLIP-based applications

#### Week 3-4: Capstone Project
- **Lecture 30:** Project Planning and Best Practices
  - Project scoping
  - Literature review
  - System design
  - Evaluation methodologies
  - Presentation skills
- **Practice 30:** Capstone Project
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
- **Quizzes:** Theory and conceptual understanding
- **Coding Assignments:** Implementation from scratch
- **Mini-Projects:** End-to-end pipelines
- **Exams:** Comprehensive understanding

### Major Assessments
- **Phase 1 Final:** Systems and programming competency
- **Phase 2 Final:** ML theory and implementation
- **Phase 3 Capstone:** Complete DL project with presentation

### Evaluation Criteria
- **Theory:** Ability to derive and explain algorithms
- **Implementation:** Code quality, correctness, efficiency
- **Analysis:** Critical thinking and error analysis
- **Communication:** Presentation and documentation skills

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

*This syllabus is designed to produce graduates who are both theoretically grounded and practically capable, ready to contribute to the ML/DL field at a professional level.*

