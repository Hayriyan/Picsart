# 15-Month Machine Learning & Deep Learning Program Syllabus

## Program Overview

**Total Duration:** 15 months \
**Structure:** 4 main phases \
**Format:** Each topic = 1 Lecture 2.5h (theory + implementation example) + 1 Practice (hands-on) \
**Primary Framework:** PyTorch \
**Secondary Framework:** TensorFlow/Keras (comparative) \
**Assessment:** Exams/quizzes, coding assignments, projects, capstone 

---

## Phase 1: Basic CS and Core Python

### Lecture 1: Introduction — What is AI, ML, DL, CS

- Historical context and evolution of AI
- Definitions: AI vs ML vs Deep Learning vs Computer Science
- Application domains (CV, NLP, Robotics, GenAI)
- Learning paradigms (supervised, unsupervised, reinforcement)
- ML problem types (classification, regression, clustering)

**Practice 1:**
- Explore AI applications across domains
- Case study analysis
- Ethical dilemma discussions

---

### Lecture 2: Computer Architecture — Programming Language, Bit & Byte

- What is a programming language (low-level vs high-level)
- How programs are compiled/interpreted
- Bits and bytes: binary representation
- Encoding systems (ASCII, UTF-8)
- Computational complexity (intuitive introduction)

**Practice 2:**
- Binary and encoding exercises
- Explore how Python code executes under the hood
- Simple algorithm implementation

---

### Lecture 3: OS Fundamentals

- What is an operating system
- Processes vs threads
- File system structure and navigation
- System calls and kernel interaction
- I/O abstraction
- Linux essentials (file system, basic commands, package management)
- Git fundamentals (init, add, commit, branches, merges)

**Practice 3:**
- Linux command-line exercises
- Process and thread management in Python
- Git workflow setup

---

### Lecture 4: Data Representation — Decimal (Int) and IEEE-754 (Float)

- Integer representation: signed/unsigned, two's complement
- Floating-point representation: IEEE-754 standard
- Precision and rounding errors in floats
- Practical consequences in numerical computing
- Type conversions

**Practice 4:**
- Floating-point precision experiments
- Integer overflow and underflow exercises
- Type conversion challenges

---

### Lecture 5: CPU and Memory Architecture

- CPU architecture fundamentals (ALU, control unit, registers)
- Memory hierarchy: registers, cache (L1/L2/L3), RAM, storage
- Cache locality and performance implications
- Storage systems (HDD vs SSD)
- How Python interacts with memory

**Practice 5:**
- Memory access pattern experiments
- Cache locality benchmarks
- Memory profiling in Python

---

### Lecture 6: GPU and Memory Architecture

- GPU architecture: cores, warps, thread blocks
- GPU memory hierarchy (global, shared, registers)
- CPU vs GPU: when to use which
- Why GPUs are essential for ML/DL
- Introduction to CUDA concepts (high level)

**Practice 6:**
- Benchmark CPU vs GPU operations
- GPU memory management basics
- Vectorized operations on CPU vs GPU

---

### Lecture 7: Numeric and String Types

- Python execution model
- Numeric types: `int`, `float`, `complex`
  - Integer operations and precision
  - Float limitations (IEEE-754 in Python)
  - Type conversions
- String type: `str`
  - Creation, manipulation, immutability
  - String methods and formatting (f-strings, `.format()`)
  - Encoding and Unicode

**Practice 7:**
- Numeric computation exercises
- String manipulation challenges
- Type conversion and formatting tasks

**Homework 7:**
- Implement a number-base converter (binary ↔ decimal ↔ hex)
- String processing pipeline (clean, parse, format)

---

### Lecture 8: Sequence Types — Lists and Tuples

- List type: `list`
  - Creation, indexing, slicing
  - List methods (append, extend, insert, remove, pop, sort)
  - List comprehensions
  - Mutability and references
- Tuple type: `tuple`
  - Creation, immutability, unpacking
  - When to use tuples vs lists
- Sequence operations (slicing, concatenation, repetition)

**Practice 8:**
- List manipulation exercises
- List comprehension challenges
- Tuple unpacking and sequence operations

**Homework 8:**
- Implement a basic stack and queue using lists
- Tuple-based record system

---

### Lecture 9: Mapping and Set Types

- Dictionary type: `dict`
  - Creation, access, methods (keys, values, items, get, update)
  - Dictionary comprehensions
  - Hashable keys and hash tables
- Set type: `set`
  - Creation, operations (union, intersection, difference)
  - Set comprehensions
  - Use cases (uniqueness, membership testing)

**Practice 9:**
- Dictionary manipulation exercises
- Set operations and algorithms
- Data structure selection challenges

**Homework 9:**
- Word frequency counter using dict
- Duplicate detection using sets

---

### Lecture 10: Python Statements — `if`, `for`, `while`

- Conditional statements (`if`, `elif`, `else`)
- `for` loops: iterating over sequences, `range()`
- `while` loops: condition-based iteration
- Loop control (`break`, `continue`, `else` clause)
- Nested loops and comprehensions

**Practice 10:**
- Control flow exercises
- Pattern printing challenges
- Nested loop problems

**Homework 10:**
- Implement FizzBuzz variants
- Build a simple text-menu CLI program

---

### Lecture 11: Function Basics — Arguments and Return Types

- Defining functions with `def`
- Positional and keyword arguments
- Default argument values
- Return values (single and multiple)
- Type hints (basic annotation)
- `*args` and `**kwargs`

**Practice 11:**
- Function implementation challenges
- Argument unpacking exercises
- Building utility functions

**Homework 11:**
- Implement a small math utilities library
- Function with flexible argument signatures

---

### Lecture 12: Function Scope — LEGB

- Local, Enclosing, Global, Built-in scope (LEGB rule)
- `global` and `nonlocal` keywords
- Namespace lookup and shadowing
- Variable lifetime

**Practice 12:**
- Scope debugging exercises
- Namespace experiments
- Identifying and fixing scope bugs

**Homework 12:**
- Scope puzzle problems
- Refactor code to avoid global state

---

### Lecture 13: Function Argument Matching Syntax, Closures

- Positional-only (`/`) and keyword-only (`*`) parameters
- Argument matching order rules
- Closures: functions capturing enclosing scope
- Factory functions using closures
- Practical closure patterns

**Practice 13:**
- Closure construction exercises
- Factory function patterns
- Argument matching edge cases

**Homework 13:**
- Implement a counter factory using closures
- Memoization using closures

---

### Lecture 14: Recursion and Higher-Order Functions

- Recursive thinking: base case and recursive case
- Call stack and recursion depth
- Higher-order functions: `map`, `filter`, `reduce`
- Lambda functions
- Functional programming patterns in Python

**Practice 14:**
- Classic recursive problems (factorial, Fibonacci, binary search)
- Higher-order function pipelines
- Lambda usage exercises

**Homework 14:**
- Implement merge sort recursively
- Build a functional data processing pipeline

---

### Lecture 15: Iterables, Iterators, and Generators

- Iterable vs iterator distinction
- Iterator protocol (`__iter__`, `__next__`)
- Generator functions (`yield` keyword)
- Generator expressions
- Memory-efficient data processing with generators
- `itertools` module overview

**Practice 15:**
- Build custom iterator classes
- Implement generator-based pipelines
- Memory-efficient data processing exercises

**Homework 15:**
- Implement a lazy file reader generator
- Custom range generator with step support

---

### Lecture 16: Files & Modules

- File I/O: reading and writing text and binary files
- File path handling (`pathlib`)
- Context managers (`with` statement)
- CSV and JSON handling
- Modules: `import` system, `__init__.py`
- Package structure and virtual environments
- `pip`, `conda`, dependency management

**Practice 16:**
- Build a data parser (CSV → JSON → cleaned output)
- Package structure exercise
- Virtual environment setup

**Homework 16:**
- Build a file-based key-value store
- Create a reusable Python package with proper structure

---

### Phase 1 Exam

**Format:** Theory Test + Coding
**Score:** 20% Intermediate Exams + 30% Homeworks + 60% Final Exam of Phase

**Theory Test:**
- CS fundamentals (architecture, memory, OS, data representation)
- Python types, control flow, functions, scope, closures, generators

**Coding Exam:**
- Implement a complete Python program from scratch
- Must demonstrate: functions, closures/generators, file I/O, modules, data structures

---

## Phase 2: Advanced Python and SQL

### Lecture 17: Introduction to Object-Oriented Analysis and Design

- Object-oriented thinking: objects, classes, responsibilities
- Classes and objects fundamentals
- Instance attributes and methods
- `__init__` constructor and `self`
- Instance vs class attributes
- OOP vs procedural programming

**Practice 17:**
- Create simple classes
- Implement basic OOP structures
- Attribute and method exercises

**Homework 17:**
- Model a real-world system (e.g., library, bank account) using classes

---

### Lecture 18: Special Methods (Getter, Setter) and Operator Overloading

- Special (dunder) methods: `__str__`, `__repr__`, `__len__`, `__eq__`, `__hash__`
- Operator overloading: `__add__`, `__sub__`, `__mul__`, `__lt__`, etc.
- Property decorators: `@property`, `@setter`, `@deleter`
- Context managers: `__enter__`, `__exit__`

**Practice 18:**
- Implement special methods on custom classes
- Create operator-overloaded numeric types
- Build a context manager

**Homework 18:**
- Implement a `Vector` class with full operator support
- Build a class with `@property`-based validation

---

### Lecture 19: Inheritance and Method Resolution

- Class inheritance and method overriding
- `super()` function
- Method Resolution Order (MRO) and `__mro__`
- Multiple inheritance
- `isinstance()` and `type()`

**Practice 19:**
- Build multi-level inheritance hierarchies
- Explore MRO with multiple inheritance
- Override and extend methods

**Homework 19:**
- Model an animal taxonomy using inheritance
- Debug an MRO conflict

---

### Lecture 20: Abstract Classes and Interfaces

- Abstract base classes (`abc` module, `ABC`, `@abstractmethod`)
- Interfaces via ABCs
- Protocols and structural subtyping (`typing.Protocol`)
- Composition vs inheritance

**Practice 20:**
- Define abstract base classes
- Implement concrete subclasses
- Use `Protocol` for duck typing

**Homework 20:**
- Design a plugin system using ABCs
- Implement the Strategy design pattern

---

### Intermediate Exam (T Exam)

**Format:** Theory Test + Coding
**Score:** 20% of Phase 2 grade

**Topics:** OOP fundamentals, special methods, inheritance, abstract classes

---

### Lecture 21: Concurrency in Python (CPU & GPU)

- Global Interpreter Lock (GIL) and its implications
- Multithreading vs multiprocessing
- CPU-bound vs I/O-bound workloads
- `threading` module
- `multiprocessing` module (Process, Pool)
- `concurrent.futures` (ThreadPoolExecutor, ProcessPoolExecutor)
- Introduction to GPU programming concepts in Python

**Practice 21:**
- Benchmark sequential vs parallel code
- Implement parallel data processing
- Threading and multiprocessing exercises

**Homework 21:**
- Parallel file processor using multiprocessing
- Thread-safe counter with locks

---

### Lecture 22: Polymorphism

- Polymorphism via inheritance (method overriding)
- Duck typing in Python
- `@staticmethod` and `@classmethod`
- Design patterns enabled by polymorphism (Factory, Strategy, Template Method)
- OOP patterns in ML systems (Dataset, Model, Trainer, Callback classes)

**Practice 22:**
- Implement polymorphic class hierarchies
- Build ML-related OOP structure (Dataset, Trainer classes)
- Design pattern exercises

**Homework 22:**
- Build a pluggable ML pipeline using polymorphism
- Implement a Callback system

---

### Lecture 23: NumPy and Vectorization

- NumPy arrays: creation, shape, dtype
- Array indexing and slicing (basic, fancy, boolean)
- Broadcasting rules
- Vectorization techniques and performance vs loops
- Memory layout (C vs Fortran order), views vs copies
- NumPy best practices for ML

**Practice 23:**
- NumPy manipulation exercises
- Vectorization vs loop comparison benchmarks
- Array operation challenges

**Homework 23:**
- Implement matrix operations from scratch using NumPy
- Vectorized implementation of distance metrics

---

### Lecture 24: Pandas and EDA

- Pandas `DataFrame` and `Series`
- Data loading (CSV, JSON, Excel)
- Data manipulation: merge, groupby, pivot, transform
- Handling missing values and duplicates
- Data visualization: Matplotlib, Seaborn
- Exploratory Data Analysis (EDA) methodology
- Statistical summaries

**Practice 24:**
- Pandas manipulation exercises
- Build a full EDA pipeline on a real dataset
- Visualization challenges

**Homework 24:**
- Complete EDA report on a provided dataset
- Feature correlation analysis with visualizations

---

### Lecture 25: Advanced GPU Architecture

- GPU memory hierarchy in depth (global, shared, L1/L2 cache, registers)
- CUDA programming model (threads, blocks, grids)
- Memory access patterns: coalesced vs strided
- Matrix multiplication on GPU
- GPU bottlenecks: memory bandwidth vs compute
- Relevance to deep learning (why matrix ops are fast on GPU)

**Practice 25:**
- GPU memory profiling
- Compare matrix multiplication: CPU NumPy vs GPU
- Identify bottlenecks in simple GPU kernels

**Homework 25:**
- Analyze performance of a matrix operation at different sizes
- Written explanation: why deep learning needs GPUs

---

### Lecture 26: Introduction to Database Management Systems

- What is a DBMS and why it matters
- Relational vs non-relational databases
- Tables, rows, columns, primary and foreign keys
- Data types in SQL
- ACID properties
- When to use a database vs flat files

**Practice 26:**
- Design an ER diagram for a simple system
- Create a schema in SQLite
- Normalization exercises

**Homework 26:**
- Design and create a normalized database for a given scenario

---

### Lecture 27: Introduction to SQL with SQLite

- `SELECT`, `FROM`, `WHERE`, `ORDER BY`, `LIMIT`
- Filtering with conditions (`AND`, `OR`, `NOT`, `IN`, `BETWEEN`, `LIKE`)
- Aggregate functions (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`)
- `GROUP BY` and `HAVING`
- Joins: `INNER`, `LEFT`, `RIGHT`, `FULL OUTER`
- Subqueries

**Practice 27:**
- SQL query exercises on a provided SQLite database
- Multi-table join problems
- Aggregation and filtering challenges

**Homework 27:**
- Write queries to answer 10 business questions from a dataset
- Implement a Python script that queries and exports results

---

### Lecture 28: Introduction to SQLite Functions

- Built-in scalar functions (string, numeric, date/time)
- `COALESCE`, `NULLIF`, `CASE WHEN`
- String functions (`SUBSTR`, `TRIM`, `REPLACE`, `UPPER`, `LOWER`)
- Date and time functions
- User-defined functions via Python (`sqlite3` module)

**Practice 28:**
- SQLite function exercises
- Python + SQLite integration
- Data cleaning using SQL functions

**Homework 28:**
- Build a Python script that uses SQLite UDFs to clean and transform a dataset

---

### Lecture 29: From Queries to Automation — Views and Triggers in SQLite

- Views: creating, using, and dropping
- When to use views vs repeated queries
- Triggers: `BEFORE`, `AFTER`, `INSTEAD OF`
- Automating data integrity with triggers
- Indexes: creation, types, performance impact

**Practice 29:**
- Create views for reporting
- Implement triggers for audit logging
- Index performance comparison

**Homework 29:**
- Build a data pipeline using views and triggers
- Performance analysis: indexed vs non-indexed queries

---

### Lecture 30: Data Visualization and EDA (Advanced)

- Review of Matplotlib and Seaborn
- Advanced plots: violin, pair plots, heatmaps, facet grids
- Interactive visualization with Plotly
- EDA checklist and methodology
- Communicating findings: report writing basics
- Connecting SQL data to Pandas and visualization

**Practice 30:**
- Full EDA project: SQL → Pandas → Visualization
- Build an interactive dashboard

**Homework 30:**
- End-to-end EDA: load from SQLite, clean in Pandas, visualize findings, write a short report

---

### Phase 2 Exam

**Format:** Project
**Score:** 30% Intermediate Exams + 30% Homeworks + 40% Final Exam of Phase

**Project Requirements:**
- Build an end-to-end data pipeline:
  - Data stored and queried from SQLite
  - Cleaned and transformed with Pandas
  - EDA with visualizations
  - NumPy-based analysis component
  - OOP-structured codebase
  - Documented and version-controlled on GitHub

---

## Phase 3: Machine Learning

### Lecture 31: Introduction to Artificial Intelligence

- AI vs ML vs Deep Learning
- Types of Machine Learning
  - Supervised
  - Unsupervised
  - Reinforcement Learning
- Regression vs Classification
- Machine Learning workflow
- Applications of ML

**Practice 31 and Homework A — EDA Milestone #1 (Easy Dataset)**
- Basic EDA
- Missing value handling
- Encoding categorical features
- Feature scaling

---

### Lecture 32: k-Nearest Neighbors & Data Splitting

- k-NN intuition
- Distance metrics
- Choosing K
- Train / Validation / Test split
- k-Fold cross-validation

**Practice 32:**
- Dataset: Wine Quality / Breast Cancer
- Tasks:
  - Apply KNN with sklearn
  - Tune K with cross-validation
  - Preprocess features

**Homework 32:**
- Implement k-NN from scratch
- Implement cross-validation manually

---

### Lecture 33: Decision Trees & Overfitting

- Tree structure
- Underfitting vs overfitting
- Depth control

**Practice 33:**
- Dataset: Lending Club dataset (tabular, larger dataset)
- Tasks:
  - Train Decision Tree
  - Visualize tree
  - Feature importance

**Homework 33:**
- Implement Decision Tree from scratch
- Overfitting experiments

---

### Lecture 34: Perceptron & Support Vector Machines (Conceptual)

- Linear classifiers
- Margin
- Kernel intuition

**Practice 34:**
- Dataset: MNIST digits subset (image)
- Tasks:
  - Train SVM with RBF kernel
  - Visualize misclassified digits

**Homework B — EDA Milestone #2 (Medium Dataset)**
- Advanced EDA
- Outlier detection
- Feature encoding

---

### Lecture 35: Linear Regression

- Model equation
- Mean Squared Error (MSE)
- Normal equation

**Practice 35:**
- Dataset: Bike sharing dataset (tabular, time-series)
- Task:

**Homework 35:**
- Implement Linear Regression from scratch

---

### Lecture 36: Generalized Linear Models & Polynomial Regression

- Feature expansion
- Non-linear regression
- Overfitting

**Practice 36:**
- Dataset: Bike sharing dataset (tabular, time-series)
- Task:

**Homework 36:**
- Polynomial regression implementation
- Overfitting visualization

---

### Lecture 37: Regularization Techniques

- L2 Regularization (Ridge)
- L1 Regularization (Lasso)
- Elastic Net (concept)
- Bias–Variance tradeoff
- Feature selection

**Practice 37:**
- Dataset: Bike sharing dataset (tabular, time-series)
- Task:

**Homework 37:**
- Ridge regression from scratch
- Basic Lasso implementation
- Compare Linear vs Polynomial vs Ridge vs Lasso

---

### Lecture 38: Metrics & Bias–Variance Tradeoff

- Regression metrics
- Classification metrics
- Confusion Matrix
- ROC and AUC
- Learning curves

**Practice 38:**
- Intro to TensorFlow

**Homework C — EDA Milestone #3 (Medium–Hard Dataset)**
- Model evaluation
- Feature engineering
- Bias–Variance analysis

---

### Lecture 39: Logistic Regression

- Sigmoid function
- Log-loss
- Decision boundary

**Practice 39:**
- Dataset: Voice dataset (audio, binary: male/female)
- Tasks:
  - Extract MFCC features
  - Train logistic regression with TensorFlow
  - Evaluate metrics

**Homework 39:**
- Implement Logistic Regression from scratch

---

### Lecture 40: Optimization Algorithms

- Gradient Descent (derivation)
- Stochastic Gradient Descent
- Mini-batch Gradient Descent
- Momentum & Nesterov
- Adam, RMSProp, AdaGrad
- Learning rate scheduling
- Second-order methods
- Non-convex optimization

**Practice 40:**
- Dataset: Blood Cell Images (Image Classification)
- Tasks:
  - Train logistic regression with TensorFlow
  - Use optimization tools

**Homework D — EDA Milestone #4 (Large Dataset)**
- Advanced EDA
- Normalization
- GD vs Adam comparison
- Implement GD & SGD

---

### Lecture 41: Advanced Support Vector Machines

- Soft margin SVM
- Kernel trick
- Hyperparameters
- Multiclass SVM

**Practice 41:**
- Dataset: Fashion-MNIST (image classification)
- Task:
  - Use SVM with soft margin

**Homework 41:**
- Kernel comparison
- Hyperparameter tuning

---

### Lecture 42: Ensemble Methods

- Decision Trees (ID3, CART, C4.5)
- Splitting criteria
- Pruning
- Random Forests
- Gradient Boosting (XGBoost / LightGBM)
- AdaBoost
- Feature importance
- Model comparison

**Practice 42:**
- Dataset: Wine quality / toy dataset
- Tasks:
  - Train Random Forest & Gradient Boosting
  - Use TensorFlow

**Homework 42:**
- Random Forest implementation
- Boosting implementation
- Feature importance report

---

### Lecture 43: Naive Bayes & Gaussian Discriminant Analysis

- Probabilistic classifiers
- Generative vs discriminative models

**Practice 43:**
- Dataset: Voice dataset (audio, binary: male/female)
- Tasks:
  - Implement Gaussian Naive Bayes using TensorFlow Probability
  - Model the distribution of acoustic features (Mean Frequency, SD)
  - Compare the decision boundary of this Generative model vs. the Logistic Regression from Practice 39

**Homework 43:**
- Implement Naive Bayes from scratch

---

### Lecture 44: Clustering — k-Means Family

- Hierarchical clustering
- k-Means
- Silhouette score
- Soft k-Means
- k-Medoids
- k-Means++

**Practice 44:**
- Dataset: Blood Cell Images / MNIST Digits
- Tasks:
  - Flatten image pixels into vectors
  - Apply k-Means to automatically group similar-looking cells/digits
  - Use the Silhouette score to evaluate cluster separation
  - Visualize the centroids (average image per cluster)

**Homework 44:**
- Implement k-Means from scratch
- Clustering evaluation

---

### Lecture 45: Gaussian Mixture Models & EM Algorithm

- Gaussian Mixture Models
- Expectation–Maximization

**Practice 45:**
- Dataset: Satellite Imagery (Land Cover)
- Tasks:
  - Use GMM for color quantization / segmentation on satellite photos
  - Group pixels into classes (water, forest, urban) by color distribution
  - Observe how soft clustering handles mixed pixels (e.g., shoreline)

**Homework 45:**
- EM algorithm implementation

---

### Lecture 46: Spectral Clustering & DBSCAN

- Density-based clustering
- Noise handling
- Anomaly detection

**Practice 46:**

**Homework E — EDA Milestone #5 (Real-World Dataset)**
- Noise handling
- Density clustering
- Anomaly detection

---

### Lecture 47: Dimensionality Reduction & Final Project

- Principal Component Analysis (PCA)
- t-SNE

**Practice 47:**

**Homework F — Final Project (Production Dataset)**
- Full end-to-end ML pipeline
- EDA
- Cleaning
- Feature engineering
- Dimensionality reduction
- Modeling
- Evaluation
- Final report

---

### Phase 3 Exam

**Format:** Project & Article
**Score:** 30% Homeworks + 60% Final Exam of Phase

**Project Requirements:**
- End-to-end ML pipeline on a real-world dataset
- Implement at least 3 algorithms from scratch
- Compare with scikit-learn / TensorFlow
- Full EDA, preprocessing, evaluation, and error analysis

**Article Requirements:**
- Written report (5–8 pages)
- Cover algorithm derivations, implementation decisions, results, and comparison with library implementations

---

## Phase 4: Deep Learning

### Lecture 48: Neural Networks

- Perceptron and multi-layer perceptron (MLP)
- Forward propagation (mathematical derivation)
- Loss functions (MSE, cross-entropy, hinge loss)
- Activation functions (sigmoid, tanh, ReLU, Leaky ReLU, ELU, Swish)
- Weight initialization (Xavier/Glorot, He initialization)
- Vanishing/exploding gradients

**Practice 48:**
- Dataset: MNIST
- Tasks:
  - Build a neural network from scratch using NumPy
  - Experiment with different activation functions
  - Compare with PyTorch implementation

**Homework 48:**
- Implement a 2-layer MLP from scratch
- Activation function comparison report

---

### Lecture 49: Backpropagation

- Computational graphs
- Chain rule derivation
- Manual backpropagation through a network
- Gradient flow and numerical gradient checking
- Gradient clipping

**Practice 49:**
- Dataset: MNIST
- Tasks:
  - Implement backpropagation manually (NumPy)
  - Verify with numerical gradient checking
  - Visualize gradient magnitudes across layers

**Homework 49:**
- Implement full forward + backward pass for a 3-layer network
- Gradient vanishing experiment with different activations

---

### Lecture 50: Convolutional Neural Networks (CNNs)

- Convolution operation (mathematical definition)
- Kernels, stride, padding
- Pooling layers (max pooling, average pooling)
- Feature maps and receptive fields
- CNN for image classification pipeline

**Practice 50:**
- Dataset: CIFAR-10
- Tasks:
  - Implement a simple CNN in PyTorch
  - Visualize learned filters
  - Compare with fully connected baseline

**Homework 50:**
- Implement a CNN from scratch in NumPy (forward pass)
- Filter visualization on a trained model

---

### Lecture 51: CNN Architectures

- LeNet, AlexNet, VGG
- GoogLeNet (Inception modules)
- ResNet (residual connections, skip connections)
- Normalization: BatchNorm, LayerNorm, GroupNorm
- Regularization: Dropout, DropConnect, Data Augmentation
- Transfer learning strategies

**Practice 51:**
- Dataset: CIFAR-10 / ImageNet subset
- Tasks:
  - Train ResNet in PyTorch
  - Implement data augmentation
  - Fine-tune a pre-trained model

**Homework 51:**
- Architecture comparison: train VGG vs ResNet, analyze results
- Transfer learning experiment with frozen vs unfrozen layers

---

### Lecture 52: Recurrent Neural Networks & LSTM

- Sequence modeling and the need for RNNs
- Vanilla RNN: architecture and forward pass
- Backpropagation Through Time (BPTT)
- Vanishing gradient in RNNs
- Long Short-Term Memory (LSTM): cell state, gates
- Gated Recurrent Unit (GRU)
- Bidirectional RNNs

**Practice 52:**
- Dataset: Time-series / text dataset
- Tasks:
  - Implement vanilla RNN in PyTorch
  - Build LSTM for sequence classification
  - Compare RNN vs LSTM on a vanishing gradient task

**Homework 52:**
- Implement LSTM cell from scratch
- Train GRU vs LSTM: performance and gradient comparison

---

### Lecture 53: Attention and Transformers

- Limitations of RNNs for long sequences
- Attention mechanism: query, key, value
- Self-attention and multi-head attention
- Positional encoding
- Transformer architecture (encoder, decoder)
- BERT and GPT: pre-training paradigms
- Fine-tuning strategies

**Practice 53:**
- Tasks:
  - Implement scaled dot-product attention from scratch
  - Build a Transformer encoder in PyTorch
  - Fine-tune a pre-trained BERT model for text classification

**Homework 53:**
- Implement multi-head attention from scratch
- Positional encoding analysis and visualization

---

### Lecture 54: Object Detection, Image Segmentation, and Visualizing CNNs

- Object detection: R-CNN, Fast R-CNN, Faster R-CNN, YOLO, SSD, DETR
- Semantic segmentation: FCN, UNet
- Instance segmentation: Mask R-CNN
- CNN visualization techniques: saliency maps, Grad-CAM, feature visualization
- Attention maps in vision models

**Practice 54:**
- Tasks:
  - Object detection with a pre-trained YOLO / Faster R-CNN model
  - Implement UNet for image segmentation
  - Generate Grad-CAM visualizations on a trained CNN

**Homework 54:**
- Grad-CAM analysis: which regions does the model focus on?
- Segmentation evaluation: IoU and Dice score implementation

---

### Lecture 55: Generative Models I — VAE & GAN

- Autoencoders: architecture and latent space
- Variational Autoencoders (VAE): mathematical formulation, ELBO derivation
- Reparameterization trick
- Generative Adversarial Networks (GANs): generator, discriminator
- GAN training dynamics and instabilities
- DCGAN architecture

**Practice 55:**
- Tasks:
  - Implement VAE from scratch in PyTorch
  - Train a DCGAN on an image dataset
  - Visualize latent space interpolation

**Homework 55:**
- VAE reconstruction quality analysis
- GAN training stability experiments (mode collapse observation)

---

### Lecture 56: Generative Models II — Diffusion Models

- Score-based generative models
- Denoising Diffusion Probabilistic Models (DDPM): forward and reverse process
- DDIM: faster sampling
- Architecture: U-Net as diffusion backbone
- Guidance techniques (classifier-free guidance)
- Stable Diffusion architecture overview

**Practice 56:**
- Tasks:
  - Train a simplified DDPM on a small image dataset
  - Compare DDPM vs DDIM sampling speed
  - Generate images with a pre-trained Stable Diffusion model

**Homework 56:**
- Implement the DDPM forward (noising) process from scratch
- Written comparison: VAE vs GAN vs Diffusion (strengths, weaknesses, use cases)

---

### Lecture 57: Generative Models III — Autoregressive and Flow-Based Models

- Autoregressive models: PixelCNN, PixelRNN
- Autoregressive language models: GPT architecture in depth
- Normalizing flows: change of variables, invertible transformations
- Energy-based models (overview)
- Evaluation metrics for generative models: FID, IS, BLEU, Perplexity

**Practice 57:**
- Tasks:
  - Implement a simple PixelCNN for image generation
  - Train a small autoregressive language model (character-level GPT)
  - Evaluate generative models using FID and perplexity

**Homework 57:**
- Character-level language model: train and analyze samples
- Generative model evaluation report across 3 model types

---

### Lecture 58: Multimodal AI I — Vision-Language Models

- Motivation: combining vision and language
- CLIP: Contrastive Language-Image Pre-training (architecture, training objective)
- Image captioning: encoder-decoder with attention
- Visual Question Answering (VQA)
- Multi-modal fusion strategies (early, late, cross-attention)
- Large-scale pre-training for multimodal models

**Practice 58:**
- Tasks:
  - Build a CLIP-based image search system
  - Fine-tune an image captioning model
  - Implement a simple VQA pipeline

**Homework 58:**
- CLIP zero-shot classification experiment
- Analysis: how does contrastive pre-training align vision and language?

---

### Lecture 59: Multimodal AI II — Advanced Multimodal Systems

- Vision-Language models at scale: LLaVA, Flamingo, GPT-4V concepts
- Audio-visual models: speech + vision fusion
- Multi-modal generation: text-to-image, image-to-text, text-to-video concepts
- Multi-modal evaluation benchmarks
- Practical considerations: data alignment, modality imbalance, fine-tuning strategies

**Practice 59:**
- Tasks:
  - Build a multimodal classifier (image + text features)
  - Experiment with a pre-trained vision-language model (e.g., LLaVA)
  - Multi-modal data preprocessing pipeline

**Homework 59:**
- Build an end-to-end multimodal pipeline (image + text → prediction)
- Comparative analysis: unimodal vs multimodal performance

---

### Phase 4 Exam

**Format:** Project
**Score:** 30% Homeworks + 70% Final Exam of Phase

**Project Requirements:**
- Complete deep learning project from conception to working solution
- Must use PyTorch as primary framework
- Choose one of the following tracks:
  - **Computer Vision:** Object detection or segmentation on a real dataset
  - **NLP / Sequence Modeling:** Transformer-based text classification or generation
  - **Generative AI:** Train and evaluate a generative model (VAE, GAN, or Diffusion)
  - **Multimodal:** Build a system combining at least two modalities

**Deliverables:**
- Clean, documented codebase on GitHub
- Final report (8–12 pages): problem, methodology, experiments, results, analysis
- Presentation (15–20 minutes) with live demo

---

## Resources and Tools

- **Languages:** Python 3.x
- **ML Framework:** PyTorch (primary), TensorFlow/Keras (secondary)
- **Data Science:** NumPy, Pandas, Matplotlib, Seaborn, Plotly
- **ML Libraries:** scikit-learn (for comparison)
- **Version Control:** Git, GitHub
- **Environment:** Linux, Conda / Venv

---

## Notes

- **Mathematics:** All mathematical prerequisites are covered in a separate syllabus running in parallel or as a prerequisite
- **Flexibility:** Some topics may be adjusted based on cohort needs
- **Depth:** Emphasis on deep understanding over breadth
- **Projects:** All projects use real-world datasets
- **Deployment:** MLOps and deployment are optional/light exposure only

---

*This syllabus is designed to produce graduates who are both theoretically grounded and practically capable, ready to contribute to the ML/DL field at a professional level.*
