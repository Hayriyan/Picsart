# 18-Month Comprehensive Machine Learning & Deep Learning Engineer Syllabus

**Target Audience:** Students with zero prior knowledge of Python, mathematics, or computer science

---

## Core Philosophy: Five Layers of Understanding

Every concept is taught across five interconnected layers:

1. **Hardware Layer** — CPU, GPU, registers, memory architecture, bit/byte-level operations
2. **Implementation Layer** — Python coding using NumPy, PyTorch, TensorFlow
3. **Mathematical Layer** — Complete theoretical understanding, derivations, proofs
4. **Systems Layer** — Real computers, servers, cloud infrastructure, databases, networking
5. **Production Layer** — Docker, Kubernetes, MLOps, monitoring, optimization

---

## Timeline Structure (18 Months - Parallel Tracks)

### **PHASE 1: FOUNDATIONS (Months 1-3)**

Three parallel tracks starting simultaneously:

---

## **TRACK A: Computer Science Fundamentals (Months 1-3)**

### Month 1: Binary & CPU Architecture

#### **1.1 Binary, Bits, and Bytes**

- **Hardware Layer:**
  - What is a bit? (electrical charge, transistor, 0/1 representation)
  - How bits are stored in DRAM and NAND flash
  - Bit operations at CPU level (AND, OR, XOR, NOT)
  - How CPUs read/write bits at nanosecond timescales
- **Implementation Layer:**
  - Python: `bin()`, `hex()`, bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`)
  - How to view binary representation of numbers
  - Bit manipulation tricks for performance
- **Mathematical Layer:**
  - Base-2 number system mathematics
  - Two's complement for negative numbers
  - Fixed-point vs floating-point representation
  - IEEE 754 floating-point standard (sign, exponent, mantissa)
- **Systems Layer:**
  - How operating systems manage bits
  - Memory addressing and byte order (big-endian vs little-endian)
  - Virtual memory and bit-level paging
- **Production Layer:**
  - Why floating-point precision matters in ML (accumulation of rounding errors)
  - int8, int16, float16, float32, float64 trade-offs for inference
  - Quantization strategies at the bit level

---

#### **1.2 CPU Architecture Fundamentals**

- **Hardware Layer:**
  - Instruction set architecture (ISA) — x86, x86-64, ARM
  - Registers (general-purpose, special-purpose)
  - Cache hierarchy (L1, L2, L3) — size, latency, bandwidth
  - Fetch-decode-execute-store cycle
  - Pipelining and superscalar execution
  - Branch prediction and cache misses
  - Memory bus and front-side bus (FSB)
- **Implementation Layer:**
  - Python: inspecting CPU details with `cpuinfo`
  - Understanding which Python operations are CPU-bound
  - Memory profiling to identify cache issues
  - SIMD operations in NumPy (vectorization)
- **Mathematical Layer:**
  - Cache miss rates and latency analysis
  - Big-O notation for algorithm analysis
  - Memory access patterns and stride
  - Amdahl's Law for parallelization speedup
- **Systems Layer:**
  - Process scheduling and context switching
  - syscalls and kernel interaction
  - CPU affinity and NUMA awareness
- **Production Layer:**
  - Why batch size affects performance
  - CPU vs GPU trade-offs
  - Optimizing inference for CPUs (AVX2, AVX-512)

---

### Month 2: GPU Architecture & Memory Systems

#### **2.1 GPU Architecture for ML**

- **Hardware Layer:**
  - GPU vs CPU design philosophy (memory bandwidth vs latency)
  - NVIDIA GPU architecture (Fermi, Maxwell, Pascal, Ampere, Hopper)
  - Streaming multiprocessors (SMs)
  - CUDA cores, tensor cores (specialized for matrix operations)
  - Warp concept (32 threads executing together)
  - Warp divergence and its impact
  - GPU memory hierarchy (registers, shared memory, global memory, host memory)
  - Memory coalescing for efficient access patterns
  - PCIe bandwidth and GPU-CPU communication overhead
- **Implementation Layer:**
  - CUDA basics — kernels, threads, blocks, grids
  - Writing first CUDA kernel (if using PyTorch custom kernels)
  - PyTorch: `.cuda()`, device placement, batch processing
  - Memory profiling with `torch.cuda.memory_summary()`
  - Identifying GPU bottlenecks
- **Mathematical Layer:**
  - Memory bandwidth calculations
  - Compute-to-bandwidth ratio (FLOPs vs memory throughput)
  - Matrix multiplication efficiency on tensor cores
  - Parallel algorithm analysis
- **Systems Layer:**
  - GPU driver architecture
  - NVIDIA's runtime vs driver API
  - PCIe generation differences (PCIe 3.0, 4.0, 5.0)
  - Multi-GPU systems and NVLink
- **Production Layer:**
  - GPU memory management for inference (batch vs streaming)
  - GPU utilization monitoring
  - Selecting right GPU for workload (H100 vs A100 vs RTX 4090)
  - Mixed precision training on GPUs

---

#### **2.2 Memory Systems Deep Dive**

- **Hardware Layer:**
  - DRAM technology (row/column access, refresh)
  - Static RAM (registers, cache)
  - Latency vs bandwidth trade-off
  - Memory controller architecture
  - Address translation and TLB (Translation Lookaside Buffer)
  - Prefetching mechanisms
- **Implementation Layer:**
  - Python memory profiling with `tracemalloc`, `memory_profiler`
  - NumPy memory layout (C-contiguous vs Fortran-contiguous)
  - Understanding `.stride()` in PyTorch for memory efficiency
  - Out-of-core processing for large datasets
- **Mathematical Layer:**
  - Memory access latency modeling
  - Roofline model for performance analysis
  - Cache hit/miss probability analysis
- **Systems Layer:**
  - Virtual memory and page tables
  - Page replacement algorithms (LRU, LFU)
  - Swap space and disk I/O
  - NUMA on multi-socket systems
- **Production Layer:**
  - Controlling memory growth in inference servers
  - Memory pooling and allocation strategies
  - Reducing per-request memory overhead

---

### Month 3: I/O Systems & Networking

#### **3.1 Disk Storage & I/O**

- **Hardware Layer:**
  - SSD vs HDD technology differences
  - NAND flash memory organization
  - Read/write performance characteristics
  - RAID configurations
  - NVMe vs SATA protocols
  - I/O bus architecture (PCIe, SATA)
- **Implementation Layer:**
  - Python file I/O and buffering
  - Reading large datasets efficiently
  - Memory mapping with `mmap`
  - Async I/O with `asyncio`
- **Mathematical Layer:**
  - Throughput and latency analysis
  - Random vs sequential access performance
- **Systems Layer:**
  - File systems (ext4, NTFS, ZFS)
  - Journaling and data consistency
  - Block allocation strategies
- **Production Layer:**
  - Data loading bottlenecks during training
  - Distributed data storage (HDFS, S3)
  - Caching strategies for inference

---

#### **3.2 Networking Basics**

- **Hardware Layer:**
  - Network interface card (NIC) architecture
  - Ethernet physical layer (twisted pair, fiber)
  - Network packet structure (header, payload, trailer)
  - DMA (Direct Memory Access) for network I/O
- **Implementation Layer:**
  - Python socket programming
  - HTTP requests and REST APIs
  - gRPC basics
- **Mathematical Layer:**
  - Bandwidth and latency calculations
  - Network protocol efficiency
- **Systems Layer:**
  - TCP/IP stack
  - OSI model (layers 1-7)
  - Routing and switching
  - DNS and service discovery
- **Production Layer:**
  - Model serving over networks
  - Inter-GPU communication (NVLink, Infiniband)
  - Distributed training network requirements

---

---

## **TRACK B: Programming Foundations (Months 2-4)**

### Month 2: Python Fundamentals

#### **2.1 Python Basics**

- **Hardware Layer:**
  - How Python bytecode executes on CPU
  - Python's Global Interpreter Lock (GIL)
  - Value vs reference semantics in memory
  - Stack frames and call stacks
- **Implementation Layer:**
  - Variables, data types (int, float, str, bool)
  - Lists, tuples, dictionaries, sets
  - Control flow (if, for, while)
  - Functions (definition, parameters, return values)
  - Variable scope and lifetime
  - String operations and methods
- **Mathematical Layer:**
  - Type systems and mathematical properties
  - Set operations
- **Systems Layer:**
  - How Python interpreter works
  - Module and package system
  - Import mechanisms
- **Production Layer:**
  - Writing efficient Python code
  - Performance profiling basics

---

#### **2.2 Object-Oriented Programming (OOP)**

- **Hardware Layer:**
  - How objects are stored in memory (vtable, instance variables)
  - Method dispatch and dynamic binding
- **Implementation Layer:**
  - Classes and objects
  - Inheritance and polymorphism
  - Encapsulation (public, private, protected)
  - Magic methods (`__init__`, `__str__`, `__call__`, etc.)
  - Properties and descriptors
  - Class methods and static methods
- **Mathematical Layer:**
  - Design patterns and mathematical abstractions
- **Systems Layer:**
  - Memory layout of objects
  - Metaclasses
- **Production Layer:**
  - Scalable code organization
  - Testing and mocking

---

### Month 3: Advanced Python

#### **3.1 Functional Programming & Metaprogramming**

- **Implementation Layer:**
  - Lambda functions and higher-order functions
  - Map, filter, reduce
  - List comprehensions and generators
  - Decorators (function and class decorators)
  - Context managers (`with` statements)
  - Metaclasses and dynamic class creation
- **Hardware Layer:**
  - How closures capture variables
  - Generator memory efficiency
- **Mathematical Layer:**
  - Functional programming theory
  - Monads and functors (advanced)
- **Production Layer:**
  - Writing reusable, composable code
  - Logging and debugging decorators

---

#### **3.2 Python Execution Model**

- **Hardware Layer:**
  - Bytecode to machine code compilation (JIT)
  - GIL and multi-threading limitations
  - Memory management and garbage collection
  - Reference counting
- **Implementation Layer:**
  - Disassembly with `dis` module
  - Tracing execution with `sys.settrace()`
  - Profiling with `cProfile` and `line_profiler`
  - Memory debugging with `guppy3`
- **Mathematical Layer:**
  - Complexity analysis of Python operations
- **Systems Layer:**
  - Threading vs multiprocessing
  - Asyncio and event loops
- **Production Layer:**
  - Performance bottleneck identification
  - Optimization techniques (Cython, Numba, ctypes)

---

### Month 4: Python Performance Optimization

#### **4.1 Performance Profiling & Optimization**

- **Implementation Layer:**
  - Timing measurements (`timeit`, `time.perf_counter()`)
  - CPU profiling (`cProfile`, `py-spy`)
  - Memory profiling (`memory_profiler`, `tracemalloc`)
  - Flame graphs interpretation
  - Benchmark best practices
- **Hardware Layer:**
  - CPU cache effects on performance
  - Branch prediction impact
  - Understanding CPU bottlenecks vs memory bottlenecks
- **Systems Layer:**
  - System-level performance monitoring
  - CPU affinity and NUMA effects
- **Production Layer:**
  - Production profiling and APM
  - Continuous benchmarking

---

#### **4.2 Vectorization & NumPy Introduction**

- **Implementation Layer:**
  - NumPy basics (arrays, dtypes, shapes)
  - Element-wise operations (broadcasting)
  - Linear algebra operations (dot, matmul)
  - Array slicing and indexing
  - Performance advantages of NumPy
- **Hardware Layer:**
  - How NumPy uses SIMD operations
  - Memory layout and striding
  - Why vectorization is faster
- **Mathematical Layer:**
  - Linear algebra operations
  - Broadcasting rules mathematically
- **Production Layer:**
  - Vectorization for inference speed

---

---

## **TRACK C: Mathematical Foundations (Months 3-6)**

### Month 3: Linear Algebra

#### **3.1 Vectors & Matrices**

- **Mathematical Layer:**
  - Vector definition and properties
  - Scalar multiplication and dot product
  - Geometric interpretation (magnitude, direction, angle)
  - Matrix definition and properties
  - Matrix-vector multiplication
  - Matrix-matrix multiplication
  - Transpose, determinant, trace
  - Special matrices (identity, diagonal, orthogonal, symmetric)
- **Implementation Layer:**
  - NumPy vector and matrix operations
  - Broadcasting in NumPy
  - Efficient matrix multiplication
  - Checking linear independence
- **Hardware Layer:**
  - How matrix multiplication maps to GPU operations
  - Memory access patterns for matrix operations
  - Tensor core utilization
- **Systems Layer:**
  - BLAS (Basic Linear Algebra Subprograms) libraries
  - LAPACK (Linear Algebra Package)
  - GPU libraries (cuBLAS)
- **Production Layer:**
  - Numerical stability in matrix operations
  - Precision trade-offs (float32 vs float64)

---

#### **3.2 Eigenvalues, Eigenvectors & Matrix Decomposition**

- **Mathematical Layer:**
  - Eigenvalue and eigenvector definition
  - Characteristic polynomial
  - Spectral theorem
  - Diagonalization
  - Singular Value Decomposition (SVD)
  - QR decomposition
  - Cholesky decomposition
  - LU decomposition
  - Eigenvalue problems (sparse, dense)
- **Implementation Layer:**
  - Computing eigenvalues with NumPy/SciPy
  - SVD and applications (dimensionality reduction, image compression)
  - Rank and condition number
  - Numerical libraries and stability
- **Hardware Layer:**
  - GPU-accelerated decompositions
  - Memory requirements for large matrices
- **Production Layer:**
  - PCA implementation using SVD
  - Detecting ill-conditioned matrices

---

### Month 4: Calculus & Optimization

#### **4.1 Derivatives & Gradients**

- **Mathematical Layer:**
  - Limits and continuity
  - Derivatives as rates of change
  - Partial derivatives and gradients
  - Directional derivatives
  - Jacobian matrix (multivariable derivatives)
  - Hessian matrix (second derivatives)
  - Lagrange multipliers
- **Implementation Layer:**
  - Numerical differentiation (finite differences)
  - Automatic differentiation (forward and reverse mode)
  - PyTorch `autograd` system
  - Computing gradients with `torch.autograd.backward()`
  - Gradient checking and verification
- **Hardware Layer:**
  - How GPUs compute gradients efficiently
  - Computation graphs and memory usage
- **Mathematical Layer:**
  - Chain rule (crucial for backpropagation)
  - Product rule and quotient rule
  - Implicit differentiation
- **Production Layer:**
  - Gradient numerical stability
  - Gradient clipping for training stability

---

#### **4.2 Optimization & Gradient Descent**

- **Mathematical Layer:**
  - Convex and non-convex optimization
  - Local minima, global minima, saddle points
  - Convexity and its implications
  - Gradient descent algorithm derivation
  - Momentum (velocity)
  - Adaptive learning rates (RMSprop, Adam)
  - Second-order methods (Newton's method, L-BFGS)
  - Stochastic gradient descent (SGD)
  - Batch vs mini-batch vs online learning
- **Implementation Layer:**
  - Implementing SGD from scratch
  - Implementing momentum optimizer
  - PyTorch optimizer API (torch.optim)
  - Learning rate schedules
  - Weight decay and regularization
- **Hardware Layer:**
  - GPU acceleration of optimization
  - Memory requirements for optimizer states (Adam stores 1st and 2nd moments)
- **Systems Layer:**
  - Distributed optimization
  - Synchronous vs asynchronous updates
- **Production Layer:**
  - Hyperparameter tuning (learning rate, momentum, weight decay)
  - Convergence monitoring

---

### Month 5: Probability & Statistics

#### **5.1 Probability Fundamentals**

- **Mathematical Layer:**
  - Sample spaces and events
  - Probability axioms
  - Conditional probability
  - Bayes' theorem (fundamental for ML)
  - Independence and conditional independence
  - Random variables (discrete and continuous)
  - Probability mass function (PMF) and probability density function (PDF)
  - Cumulative distribution function (CDF)
- **Implementation Layer:**
  - Sampling from distributions with NumPy/SciPy
  - Bayes' theorem calculation
  - Joint and marginal distributions
  - Naive Bayes classifier implementation
- **Mathematical Layer:**
  - Law of total probability
  - Law of large numbers
  - Central limit theorem
- **Production Layer:**
  - Bayesian inference in ML
  - Uncertainty quantification

---

#### **5.2 Distributions & Statistics**

- **Mathematical Layer:**
  - Bernoulli distribution
  - Binomial distribution
  - Gaussian (normal) distribution
  - Exponential distribution
  - Poisson distribution
  - Beta distribution
  - Categorical distribution
  - Expectation and variance
  - Covariance and correlation
  - Skewness and kurtosis
- **Implementation Layer:**
  - Statistical distributions in SciPy
  - Hypothesis testing (t-test, chi-square, ANOVA)
  - Confidence intervals
  - Maximum likelihood estimation (MLE)
  - Bayesian parameter estimation
- **Mathematical Layer:**
  - Moment-based analysis
  - Law of total expectation
- **Production Layer:**
  - Prior selection in Bayesian methods
  - Posterior inference

---

### Month 6: Information Theory & Advanced Topics

#### **6.1 Information Theory**

- **Mathematical Layer:**
  - Entropy as information content
  - Shannon's entropy formula
  - Cross-entropy
  - Kullback-Leibler (KL) divergence
  - Mutual information
  - Jensen-Shannon divergence
  - Information gain
- **Implementation Layer:**
  - Computing entropy and KL divergence
  - Cross-entropy loss in PyTorch
  - Applications to classification
- **Mathematical Layer:**
  - Derivations of entropy properties
  - Relationship between entropy and probability
- **Production Layer:**
  - KL divergence for model evaluation
  - Information-theoretic analysis of compression

---

#### **6.2 Advanced Topics: Summary & Integration**

- Connecting linear algebra to calculus
- Connecting probability to optimization
- Connecting information theory to classification
- Problem-based learning with mathematical concepts

---

---

## **PHASE 2: CLASSICAL MACHINE LEARNING (Months 4-9)**

### Month 4: Regression

#### **4.1 Linear Regression**

- **Mathematical Layer:**
  - Linear regression model: $y = \mathbf{w}^T \mathbf{x} + b$
  - Loss function: Mean Squared Error (MSE) = $\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
  - Closed-form solution (Normal Equation): $\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$
  - Gradient-based solution (SGD)
  - Bias-variance decomposition
  - Assumptions: linearity, homoscedasticity, independence
- **Implementation Layer:**
  - Implementing linear regression from scratch (NumPy)
  - Using scikit-learn's `LinearRegression`
  - PyTorch implementation with `nn.Linear` and `nn.MSELoss`
  - Loading data and train-test split
- **Hardware Layer:**
  - Computational complexity of matrix inversion vs SGD
  - Memory requirements for storing covariance matrix
  - GPU acceleration with PyTorch
- **Systems Layer:**
  - Data preprocessing and normalization
  - Handling large datasets (mini-batch gradient descent)
- **Production Layer:**
  - Model serialization (pickle, joblib, ONNX)
  - Inference latency on CPU vs GPU
  - Handling new data in production

---

#### **4.2 Polynomial & Regularized Regression**

- **Mathematical Layer:**
  - Polynomial regression: adding polynomial features
  - Overfitting phenomenon
  - L2 regularization (Ridge): $\text{MSE} + \lambda \sum_{j=1}^{d} w_j^2$
  - L1 regularization (Lasso): $\text{MSE} + \lambda \sum_{j=1}^{d} |w_j|$
  - Elastic Net: $\text{MSE} + \lambda_1 \sum |w_j| + \lambda_2 \sum w_j^2$
  - Ridge regressor solution (closed form with regularization)
  - Regularization path (how coefficients change with $\lambda$)
- **Implementation Layer:**
  - Feature engineering (polynomial features with `PolynomialFeatures`)
  - Ridge and Lasso regression in scikit-learn
  - Hyperparameter tuning with cross-validation
  - PyTorch: weight decay implements L2 regularization
- **Hardware Layer:**
  - Memory and computation with increased features
- **Systems Layer:**
  - Cross-validation strategies for hyperparameter selection
  - Train-validation-test splits
- **Production Layer:**
  - Regularization for generalization
  - Preventing overfitting in production

---

### Month 5: Classification

#### **5.1 Logistic Regression & Classification Basics**

- **Mathematical Layer:**
  - Logistic function (sigmoid): $\sigma(z) = \frac{1}{1 + e^{-z}}$
  - Logistic regression model: $P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$
  - Binary cross-entropy loss: $-\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$
  - Optimization via gradient descent
  - Softmax for multiclass: $\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_k e^{z_k}}$
  - Categorical cross-entropy for multiclass
- **Implementation Layer:**
  - Implementing logistic regression from scratch
  - scikit-learn: `LogisticRegression`
  - PyTorch: `nn.BCEWithLogitsLoss`, `nn.CrossEntropyLoss`
  - Probability calibration
- **Hardware Layer:**
  - GPU acceleration for multiclass classification
- **Systems Layer:**
  - Decision thresholds and trade-offs
  - Class imbalance handling
- **Production Layer:**
  - Threshold selection for business requirements
  - Operating point selection (ROC curve)

---

#### **5.2 Decision Trees & Ensemble Methods**

- **Mathematical Layer:**
  - Information gain and entropy-based splitting
  - Gini impurity: $\text{Gini}(p) = 1 - \sum_{i=1}^{c} p_i^2$
  - Tree growing (recursive partitioning)
  - Pruning for overfitting prevention
- **Implementation Layer:**
  - Implementing simple decision tree from scratch
  - scikit-learn: `DecisionTreeClassifier`
  - Understanding tree structure and feature importance
  - Hyperparameters: max_depth, min_samples_split, min_samples_leaf
- **Mathematical Layer:**
  - Random Forests: Bootstrap aggregating (bagging)
  - Gradient Boosting: Sequential tree fitting to residuals
  - XGBoost: Regularized gradient boosting
  - AdaBoost: Weighted ensemble
  - Voting and stacking
- **Implementation Layer:**
  - scikit-learn: `RandomForestClassifier`, `GradientBoostingClassifier`
  - XGBoost and LightGBM libraries
  - Ensemble techniques for combining models
  - Feature importance from tree-based models
- **Hardware Layer:**
  - Parallel tree growing on GPUs
  - Memory requirements for large ensembles
- **Systems Layer:**
  - Distributed decision tree training
- **Production Layer:**
  - Tree-based models in production (fast inference)
  - Model compression via pruning

---

### Month 6: Clustering & Dimensionality Reduction

#### **6.1 Clustering Algorithms**

- **Mathematical Layer:**
  - K-means objective: minimize $\sum_{i=1}^{n} \sum_{k=1}^{K} r_{ik} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$
  - K-means++ initialization
  - Hierarchical clustering (linkage criteria: single, complete, average, Ward)
  - Dendrogram interpretation
  - DBSCAN (density-based): $\varepsilon$ and $\text{min\_samples}$ parameters
  - Gaussian Mixture Models (GMM) and EM algorithm
- **Implementation Layer:**
  - K-means from scratch (NumPy)
  - scikit-learn clustering (KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture)
  - Silhouette score and Davies-Bouldin index
  - Elbow method for K selection
- **Hardware Layer:**
  - GPU acceleration for K-means
- **Production Layer:**
  - Clustering for customer segmentation
  - Scalability to billions of points

---

#### **6.2 Dimensionality Reduction**

- **Mathematical Layer:**
  - Principal Component Analysis (PCA): finding directions of maximum variance
  - PCA algorithm: eigendecomposition of covariance matrix
  - Explained variance ratio
  - t-SNE: minimizing KL divergence between high-dim and low-dim distributions
  - UMAP: manifold learning with topological preservation
  - Autoencoders as nonlinear dimensionality reduction
- **Implementation Layer:**
  - PCA from scratch using SVD
  - scikit-learn: `PCA`, `TSNE`, `UMAP`
  - Choosing number of components
  - Visualization of high-dimensional data
- **Hardware Layer:**
  - SVD computational complexity
  - GPU acceleration for matrix operations
- **Production Layer:**
  - Feature compression for faster inference
  - Computational trade-offs in inference

---

### Month 7: Feature Engineering & Model Evaluation

#### **7.1 Feature Engineering**

- **Implementation Layer:**
  - Normalization vs standardization (MinMaxScaler vs StandardScaler)
  - One-hot encoding for categorical features
  - Label encoding
  - Feature scaling impact on distance-based algorithms
  - Polynomial features and interaction terms
  - Domain-specific feature engineering
  - Binning and discretization
- **Mathematical Layer:**
  - Why standardization helps gradient descent
  - Distance metrics affected by scale
- **Systems Layer:**
  - Feature stores (Feast, Tecton)
  - Feature pipeline management
- **Production Layer:**
  - Consistency between training and inference preprocessing
  - Preventing data leakage
  - Transformation pipelines (sklearn.pipeline)

---

#### **7.2 Model Evaluation & Selection**

- **Mathematical Layer:**
  - Train-test split and cross-validation
  - K-fold CV: reduces variance in evaluation
  - Stratified CV for imbalanced datasets
  - Confusion matrix: TP, TN, FP, FN
  - Accuracy, Precision, Recall, F1-score
  - ROC curve and AUC: trade-off between TPR and FPR
  - Precision-Recall curve (better for imbalanced data)
  - Matthews Correlation Coefficient (MCC)
  - Regression metrics: MAE, RMSE, R² score, MAPE
- **Implementation Layer:**
  - Cross-validation with `sklearn.model_selection.cross_val_score`
  - Confusion matrix visualization
  - scikit-learn: `classification_report`, `confusion_matrix`
  - ROC and PR curves with scikit-learn
  - GridSearchCV and RandomizedSearchCV for hyperparameter tuning
- **Systems Layer:**
  - Nested cross-validation (outer for evaluation, inner for hyperparameter tuning)
  - Time-series cross-validation for temporal data
- **Production Layer:**
  - A/B testing models in production
  - Monitoring model performance degradation
  - Business-relevant metrics (profit, conversion, etc.)

---

### Month 8-9: Practical ML Systems

#### **8.1 Real-World ML Pipelines**

- **Systems Layer:**
  - ETL (Extract, Transform, Load) workflows
  - Data validation and quality checks
  - Missing value imputation strategies
  - Outlier detection and handling
  - Class imbalance techniques (oversampling, undersampling, SMOTE)
- **Implementation Layer:**
  - Building end-to-end pipelines with scikit-learn Pipelines
  - ColumnTransformer for different feature types
  - Reproducibility with random seeds
  - Logging and experiment tracking
- **Production Layer:**
  - Model versioning and reproducibility
  - Feature versioning
  - Retraining schedules

---

#### **8.2 Capstone Project 1: Classical ML**

**Building an end-to-end ML system:**

1. **Problem Definition:**
   - Real-world dataset (e.g., house price prediction, customer churn)
   - Business metrics and evaluation criteria

2. **Exploratory Data Analysis (EDA):**
   - Statistical summaries
   - Distribution analysis
   - Correlation and feature relationships
   - Visualizations (histograms, scatter plots, heatmaps)
   - Missing data patterns
   - Outlier identification

3. **Data Preprocessing:**
   - Handling missing values (imputation strategies)
   - Feature scaling/normalization
   - Categorical feature encoding
   - Outlier treatment
   - Feature engineering domain-specific features

4. **Model Development:**
   - Try multiple algorithms (linear models, tree-based, ensemble)
   - Cross-validation for evaluation
   - Hyperparameter tuning with GridSearch
   - Feature importance analysis
   - Ensemble methods combining multiple models

5. **Evaluation & Analysis:**
   - Test set performance
   - Error analysis (residuals, prediction patterns)
   - Learning curves (bias-variance analysis)
   - Feature importance interpretation
   - Business impact of model decisions

6. **Production Deployment:**
   - Model serialization
   - Simple REST API with Flask/FastAPI
   - Docker containerization
   - Basic monitoring (model performance tracking)
   - A/B testing framework (conceptually)

7. **Documentation:**
   - Data dictionary
   - Model card
   - Deployment guide
   - Known limitations

---

---

## **PHASE 3: MLOps & SYSTEMS FUNDAMENTALS (Months 4-9 - Parallel)**

### Month 4: Linux & Command Line

#### **4.1 Linux Fundamentals**

- **Hardware Layer:**
  - Kernel and user space
  - Process management (fork, exec, wait)
  - Memory management (virtual memory, paging)
  - Interrupt handling
- **Implementation Layer:**
  - Basic commands: ls, cd, mkdir, rm, cp, mv, find
  - Permissions: chmod, chown, umask
  - File redirection: >, >>, <, pipes |
  - Text processing: grep, sed, awk
  - Process management: ps, top, kill, bg, fg
  - File systems: inode structure, hard links, symbolic links
- **Systems Layer:**
  - Shell scripting basics (bash, zsh)
  - Task scheduling with cron
  - User and group management
  - Disk and partition management
- **Production Layer:**
  - Scripting for deployment automation
  - Log file analysis and troubleshooting

---

#### **4.2 Linux for ML Engineers**

- **Implementation Layer:**
  - Environment variables
  - Virtual environments (venv, Conda)
  - Package management (apt, brew, conda)
  - SSH and remote connections
  - Disk usage monitoring (df, du)
  - System monitoring (vmstat, iostat)
- **Production Layer:**
  - Setting up Linux servers
  - User access control for teams
  - Automated backups and snapshots

---

### Month 5: SQL & Databases

#### **5.1 SQL Fundamentals**

- **Mathematical Layer:**
  - Relational algebra (selection, projection, join)
  - Set operations (union, intersection, difference)
  - Aggregation operations
- **Implementation Layer:**
  - SELECT queries (WHERE, ORDER BY, GROUP BY, HAVING)
  - JOINs (INNER, LEFT, RIGHT, FULL OUTER)
  - Subqueries and CTEs (Common Table Expressions)
  - Aggregation functions (COUNT, SUM, AVG, MIN, MAX)
  - Window functions (ROW_NUMBER, RANK, LAG, LEAD)
  - INSERT, UPDATE, DELETE operations
  - Transactions and ACID properties
- **Systems Layer:**
  - Database design (normalization, ER models)
  - Indexing for query optimization
  - Query execution plans (EXPLAIN)
  - Table partitioning
- **Production Layer:**
  - Performance tuning SQL queries
  - Monitoring database health

---

#### **5.2 Databases for ML**

- **Implementation Layer:**
  - PostgreSQL basics
  - MongoDB (document stores)
  - Connecting to databases from Python (psycopg2, SQLAlchemy)
  - Pandas integration with SQL (pd.read_sql, to_sql)
- **Systems Layer:**
  - Data warehousing concepts
  - OLTP vs OLAP
  - Data lakes
- **Production Layer:**
  - Handling large-scale data queries
  - Data freshness and synchronization
  - Backup and disaster recovery

---

### Month 6: Version Control with Git

#### **6.1 Git Fundamentals**

- **Implementation Layer:**
  - Git basics: repository, commits, branches
  - Staging area and commits
  - History and log
  - Branches and merging
  - Conflict resolution
  - Remote repositories (push, pull, fetch)
  - GitHub/GitLab workflows
- **Systems Layer:**
  - Internal structure of Git (objects, refs)
  - Performance on large files
- **Production Layer:**
  - Branching strategies (Git Flow, GitHub Flow)
  - Code review workflows
  - CI/CD integration

---

### Month 7: Docker & Containerization

#### **7.1 Docker Basics**

- **Hardware Layer:**
  - Containers vs virtual machines
  - Linux namespaces and cgroups
  - Copy-on-write filesystem
- **Implementation Layer:**
  - Docker concepts: images, containers, registries
  - Dockerfile syntax (FROM, RUN, COPY, ENV, CMD, ENTRYPOINT)
  - Building images
  - Running containers with port and volume mapping
  - Environment variables and secrets
  - Multi-stage builds for optimization
- **Systems Layer:**
  - Container networking (bridge, host, overlay networks)
  - Volume management (bind mounts, named volumes)
  - Docker compose for multi-container applications
- **Production Layer:**
  - Creating minimal images (scratch, alpine base)
  - Image security scanning
  - Registry management (Docker Hub, ECR, GCR)
  - Container logging and monitoring

---

#### **7.2 Docker for ML Applications**

- **Implementation Layer:**
  - Dockerfile for ML models
  - Reproducible environments (pinned dependencies)
  - Development vs production images
  - Caching layers appropriately for build speed
- **Production Layer:**
  - Model serving in containers
  - GPU support in containers (nvidia-docker)
  - Scaling containers (simple orchestration)

---

### Month 8: Kubernetes Basics

#### **8.1 Kubernetes Architecture**

- **Systems Layer:**
  - Master/control plane components: etcd, API server, scheduler, controller-manager
  - Worker nodes and kubelet
  - Container runtime interface (CRI)
  - Networking between nodes (networking layer)
  - cluster DNS (coredns)
- **Implementation Layer:**
  - kubectl commands and configuration
  - YAML manifest files
  - Pods (smallest Kubernetes unit)
  - Services (exposing pods)
  - Deployments (managing pod replicas)
  - StatefulSets for stateful applications
  - DaemonSets for node-level processes
- **Production Layer:**
  - Health checks (liveness, readiness probes)
  - Resource requests and limits
  - Horizontal Pod Autoscaling (HPA)
  - Vertical Pod Autoscaling (VPA)

---

#### **8.2 Kubernetes for ML**

- **Implementation Layer:**
  - Deployment manifests for models
  - ConfigMaps and Secrets management
  - Job and CronJob for batch processing
  - Persistent volumes for data
- **Production Layer:**
  - Model serving at scale
  - Multi-replicas of models
  - Rolling updates and canary deployments
  - Distributed training orchestration

---

### Month 9: CI/CD Pipelines

#### **9.1 CI/CD Concepts**

- **Systems Layer:**
  - Continuous Integration: automatically test code changes
  - Continuous Deployment: automatically deploy to production
  - Continuous Delivery: automated releases after approval
  - Pipeline stages (build, test, deploy)
  - Triggering on events (push, pull request, schedule)
- **Implementation Layer:**
  - GitHub Actions workflows
  - GitLab CI/CD
  - Jenkins overview
  - Docker image building in CI
  - Running tests in CI
- **Production Layer:**
  - Security scanning in CI (dependency checks, SAST/DAST)
  - Approval gates
  - Rollback strategies
  - Environment promotion (dev → staging → production)

---

#### **9.2 CI/CD for ML**

- **Implementation Layer:**
  - Testing ML code (unit tests, integration tests)
  - Data validation pipelines
  - Model evaluation in CI
  - Automatic model registry updates
- **Production Layer:**
  - Automated retraining pipelines
  - Model performance gates (only deploy if better than baseline)
  - A/B testing automation

---

---

## **PHASE 4: DEEP LEARNING FUNDAMENTALS (Months 10-18)**

### Month 10: Neural Network Basics

#### **10.1 Feedforward Neural Networks**

- **Mathematical Layer:**
  - Single neuron: $y = \sigma(\mathbf{w}^T \mathbf{x} + b)$
  - Activation functions: ReLU $\max(0, x)$, Sigmoid $\frac{1}{1+e^{-x}}$, Tanh $\frac{e^x - e^{-x}}{e^x + e^{-x}}$, GELU
  - Universal approximation theorem
  - Network depth vs width trade-offs
  - Backpropagation: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \sigma} \frac{\partial \sigma}{\partial z} \frac{\partial z}{\partial w}$ (chain rule)
  - Gradient flow through layers
- **Implementation Layer:**
  - Implementing forward pass (NumPy)
  - Implementing backpropagation from scratch
  - PyTorch: `nn.Module`, `nn.Linear`, `nn.ReLU`, loss functions
  - Training loop: forward, backward, optimizer step
  - Batch processing
- **Hardware Layer:**
  - Computation graph and automatic differentiation on GPU
  - Memory requirements for storing activations (backpropagation)
  - Batch size trade-offs (memory vs gradient noise)
- **Systems Layer:**
  - Mini-batch gradient descent scaling
  - Data loading and batching
- **Production Layer:**
  - Training stability monitoring
  - Gradient clipping for stability
  - Mixed precision training (float32 for gradients, float16 for forward pass)

---

#### **10.2 Training Techniques & Optimization**

- **Mathematical Layer:**
  - Gradient Descent variants: SGD, Momentum, Adam, AdamW, RMSprop
  - Exponential moving average (momentum)
  - Adaptive learning rates (per-parameter scaling)
  - Weight decay vs L2 regularization (different in the context of momentum)
  - Learning rate scheduling (constant, decay, cyclical, warmup)
- **Implementation Layer:**
  - PyTorch optimizers (torch.optim)
  - Learning rate schedules (torch.optim.lr_scheduler)
  - Parameter groups for different learning rates
  - Gradient accumulation for larger effective batch sizes
- **Hardware Layer:**
  - GPU memory requirements for optimizer states (Adam stores first and second moments)
  - Distributed optimization across GPUs
- **Production Layer:**
  - Finding optimal learning rates with learning rate finder
  - Warmup for stable training
  - Fine-tuning strategies (lower LR for pretrained layers)

---

#### **10.3 Regularization & Generalization**

- **Mathematical Layer:**
  - Overfitting and underfitting
  - Regularization: L1, L2, elastic net
  - Dropout: randomly zeroing activations during training
  - Batch normalization: normalizing layer inputs to have mean 0, variance 1
  - Layer normalization: normalization per feature per sample
  - Early stopping: monitoring validation loss
- **Implementation Layer:**
  - `nn.Dropout`, `nn.BatchNorm1d`, `nn.BatchNorm2d`, `nn.LayerNorm`
  - Early stopping implementation
  - Weight decay in optimizers
  - Data augmentation for generalization
- **Hardware Layer:**
  - Batch normalization efficiency on GPU
  - Impact on backpropagation
- **Production Layer:**
  - Dropout disabled at inference time
  - Batch norm running statistics at inference
  - Ensemble by average predictions (approximates multiple dropouts)

---

### Month 11: Convolutional Neural Networks

#### **11.1 Convolution Operation & CNN Basics**

- **Hardware Layer:**
  - How convolution maps to GPU tensor operations
  - Memory access patterns and cache efficiency
  - GEMM-based convolution implementation
  - Memory requirements for feature maps
- **Mathematical Layer:**
  - Convolution operation: sliding window with weights
  - Convolution formula: $y[i,j] = \sum_{a,b} w[a,b] \cdot x[i+a, j+b] + b$
  - Padding: zero-padding, same padding
  - Stride: step size for sliding window
  - Dilation (atrous convolution)
  - Output size formula: $\text{out} = \frac{\text{in} + 2p - k}{s} + 1$
  - Receptive field and how it grows with depth
  - Backpropagation through convolution (transposed convolution)
- **Implementation Layer:**
  - Convolution from scratch (inefficient, for understanding)
  - PyTorch: `nn.Conv2d`, padding, stride, dilation
  - Learnable parameters in convolution layers
  - Visualizing learned filters
  - Pooling layers: max pooling, average pooling
- **Production Layer:**
  - Memory efficiency: grouped convolution, depthwise separable convolution
  - Inference optimization: im2col transformation
  - Quantization impact on convolution layers

---

#### **11.2 CNN Architectures**

- **Mathematical Layer & Implementation Layer:**

**LeNet-5:**

- Simple architecture demonstrated on MNIST
- 2 convolutional layers + 3 fully connected layers
- Foundation for modern CNNs

**AlexNet:**

- ImageNet breakthrough (2012)
- 5 convolutional layers + 3 fully connected layers
- ReLU activation (faster than sigmoid)
- GPU training (requires GPU)
- Local response normalization
- Implementation: PyTorch models or from scratch

**VGG:**

- Small 3×3 filters, stacked for large receptive fields
- Very deep architecture (16, 19 layers)
- Uniform architecture making it intuitive
- High memory consumption
- Implementation: understanding depth impact

**ResNet:**

- Residual connections: $y = f(x) + x$
- Solves vanishing gradient problem in very deep networks
- Enables training of 100+ layer networks
- Bottleneck architecture for efficiency
- Identity mappings and initialization
- Implementation: building residual blocks

**Inception/GoogLeNet:**

- Multi-scale feature extraction
- 1×1 convolutions for dimensionality reduction
- Inception modules
- Information in parallel branches at different scales

**MobileNet:**

- Depthwise separable convolution: separates spatial and channel operations
- Drastically fewer parameters (mobile-friendly)
- Width and resolution multipliers for accuracy-efficiency trade-off
- Implementation: efficient architecture design

**EfficientNet:**

- Compound scaling: balance depth, width, resolution
- Achieves SOTA with fewer parameters
- Automated scaling rules (EfficientNet-B0 through B7)

- **Hardware Layer:**
  - Parameter counts and memory requirements
  - FLOPs (floating-point operations) for each architecture
  - Inference speed on different hardware (CPU, mobile GPU, edge)
  - Memory vs accuracy trade-offs
- **Production Layer:**
  - Transfer learning with pretrained models
  - Fine-tuning strategies
  - Model selection based on latency requirements (lightweight vs accurate)

---

#### **11.3 Object Detection**

- **Mathematical Layer:**
  - YOLO: grid-based detection with bounding box regression
  - Anchor boxes and prior knowledge
  - Intersection over Union (IoU): $\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$
  - Non-maximum suppression (NMS)
  - Faster R-CNN: region proposals with RPN
  - SSD: multi-scale feature pyramids
- **Implementation Layer:**
  - Detection metrics: mAP (mean Average Precision)
  - Bounding box formats (xyxy, xywh, relative coordinates)
  - Anchor box design and generation
  - Loss functions for object detection
  - Using pretrained detectors (PyTorch Hub, TorchVision)
- **Production Layer:**
  - Real-time inference optimization
  - Handling variable input sizes
  - Deployment on edge devices
  - Post-processing and filtering detections

---

#### **11.4 Semantic & Instance Segmentation**

- **Mathematical Layer:**
  - FCN (Fully Convolutional Networks): all convolutions, transposed convolutions
  - Skip connections for precise localization
  - U-Net: encoder-decoder architecture with lateral connections
  - DeepLab: dilated convolution for large receptive field without pooling
  - Atrous spatial pyramid pooling (ASPP)
  - Mask R-CNN: instance segmentation with mask head
- **Implementation Layer:**
  - Pixel-wise classification
  - Segmentation metrics: IoU (Intersection over Union), Dice coefficient
  - Using pretrained segmentation models
  - Custom segmentation with PyTorch
- **Hardware Layer:**
  - Memory for feature maps in high resolution
  - Inference optimization with resolution trade-offs
- **Production Layer:**
  - Real-time segmentation inference
  - Medical image segmentation applications

---

### Month 12: Recurrent Neural Networks

#### **12.1 RNN Fundamentals**

- **Mathematical Layer:**
  - Recurrent connection: $h_t = f(h_{t-1}, x_t)$
  - Unrolling through time (Backpropagation Through Time - BPTT)
  - Gradient flow through time steps
  - Vanishing gradient problem: gradients shrink exponentially with time
  - Exploding gradient problem: gradients grow exponentially
  - Numerical instability
- **Implementation Layer:**
  - Implementing vanilla RNN from scratch
  - PyTorch: `nn.RNN`, `nn.RNNCell`
  - Bidirectional RNNs
  - Stacked RNNs
  - Gradient clipping for stability
- **Hardware Layer:**
  - Sequential nature makes GPU batching less efficient
  - Memory requirements for hidden states
  - Computational speed of RNNs vs CNNs
- **Production Layer:**
  - Why transformers are preferred (parallelizable)
  - Legacy RNN deployment

---

#### **12.2 LSTM & GRU**

- **Mathematical Layer:**
  - LSTM gates: forget gate, input gate, output gate, cell state
  - Forget gate: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
  - Input gate: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
  - Candidate cell state: $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
  - Cell state update: $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
  - Output gate: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
  - Output: $h_t = o_t * \tanh(C_t)$
  - GRU: simplified LSTM with reset and update gates
  - Why LSTM/GRU solve vanishing gradient: additive cell state update
- **Implementation Layer:**
  - PyTorch: `nn.LSTM`, `nn.GRU`
  - Hidden state and cell state initialization
  - Multi-layer LSTM
  - Understanding gate outputs (visualization)
- **Hardware Layer:**
  - More parameters than vanilla RNN (4x for LSTM)
  - Efficiency on GPU (less so than attention)
- **Production Layer:**
  - LSTM for sequence modeling
  - Encoder-decoder with attention for seq2seq tasks

---

#### **12.3 Attention Mechanism**

- **Mathematical Layer:**
  - Attention concept: focus on relevant input elements
  - Scaled dot-product attention: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
  - Query, Key, Value projections ($W_Q$, $W_K$, $W_V$)
  - Why scaling by $\sqrt{d_k}$: stable gradients
  - Multi-head attention: multiple attention subspaces
  - Positional encoding: absolute positions, relative positions
- **Implementation Layer:**
  - Implementing attention from scratch
  - PyTorch: `nn.MultiheadAttention`
  - Attention visualization (looking at which tokens it pays attention to)
  - Sequence-to-sequence with attention
- **Hardware Layer:**
  - Attention has quadratic complexity in sequence length
  - GPU efficiency for matrix operations
  - Memory requirements for attention matrices
- **Production Layer:**
  - Efficient attention variants (sparse attention, local attention)
  - Inference optimization

---

### Month 13: Transformer Architecture

#### **13.1 Transformer Fundamentals**

- **Mathematical Layer:**
  - Encoder-decoder architecture (self-attention in both)
  - Self-attention: queries, keys, values from same token sequence
  - Cross-attention: keys, values from encoder, queries from decoder
  - Multi-head attention mechanism
  - Feed-forward networks: $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$
  - Layer normalization and residual connections
  - Positional encoding: sinusoidal positional encoding
  - Absolute position vs relative position encodings
- **Implementation Layer:**
  - Transformer block: attention + FFN + residual + normalization
  - Building full transformer from scratch
  - PyTorch: `nn.TransformerEncoder`, `nn.TransformerDecoder`
  - Positional encoding implementation
  - Attention mask (preventing looking at future tokens in decoder)
- **Hardware Layer:**
  - Parallelizable (all tokens can be processed in parallel)
  - Quadratic memory in sequence length
  - GPU efficiency with large batch sizes
- **Production Layer:**
  - Training efficiency compared to RNNs
  - Inference optimization strategies

---

#### **13.2 Vision Transformers (ViT)**

- **Mathematical Layer:**
  - Splitting image into patches
  - Patch embedding
  - Concatenating class token
- **Implementation Layer:**
  - Patch embedding layer
  - Vision transformer architecture
  - Fine-tuning on downstream tasks
  - Comparison with CNNs
- **Production Layer:**
  - Transfer learning with vision transformers
  - Efficiency on different hardware

---

#### **13.3 Transformer Optimization**

- **Mathematical Layer & Implementation Layer:**

**Flash Attention:**

- Reducing memory accesses to GPU memory
- Attention in blocks with recomputation
- Significant speedup and memory savings

**Grouped Query Attention (GQA):**

- Multiple query projections share key/value
- Reduce memory requirements
- Faster inference with less parameter overhead

**Sparse Attention:**

- Full attention is quadratic; sparse patterns reduce cost
- Strided attention, local attention, combinations
- Lossing some connectivity for efficiency

- **Production Layer:**
  - Inference speed optimization
  - Reduced latency APIs
  - Serving large models efficiently

---

### Month 14: Large Language Models

#### **14.1 LLM Basics & Tokenization**

- **Mathematical Layer & Implementation Layer:**

**Tokenization:**

- Byte Pair Encoding (BPE): iteratively merging most frequent byte pairs
- SentencePiece: unigram language model for tokenization
- WordPiece: similar to BPE, used by BERT
- Vocabulary size trade-offs
- Handling out-of-vocabulary tokens and special tokens

**Language Modeling Objective:**

- Causal language modeling (GPT): predict next token
- Masked language modeling (BERT): predict masked tokens
- Contrastive learning (SimCLR-like approaches)

- **Implementation Layer:**
  - Using transformers library for tokenizers
  - Creating custom tokenizers with tokenizers library
  - Tokenization consistency across training/inference
  - Context window size effects
- **Production Layer:**
  - Vocabulary size vs model quality
  - Tokenizer consistency important for reproducibility

---

#### **14.2 Popular LLM Models**

- **Mathematical Layer & Implementation Layer:**

**BERT:**

- Bidirectional encoding
- Masked language modeling objective
- Segment embeddings for multi-sentence understanding
- Applications: classification, NLU tasks
- Fine-tuning on downstream tasks

**GPT-2/GPT-3:**

- Unidirectional (left-to-right) language modeling
- Autoregressive generation
- Few-shot learning capabilities (GPT-3)
- Scaling laws (more parameters, more data, better performance)

**GPT-4:**

- Improved reasoning, reduced hallucinations
- Multimodal (text + images)
- RLHF (Reinforcement Learning from Human Feedback)

**LLaMA/Mistral:**

- Open-source alternatives
- Efficient architectures (e.g., rotary positional embeddings)
- Competitive with proprietary models

**Gemini:**

- Multimodal architecture
- Efficient scaling

---

#### **14.3 Fine-tuning & Adaptation**

- **Mathematical Layer:**
  - Full model fine-tuning
  - Transfer learning: adjusting pretrained weights
  - Catastrophic forgetting: loss of pretrained knowledge during fine-tuning
- **Implementation Layer:**

**Full Fine-tuning:**

- All weights trainable
- Highest quality but most expensive
- Large memory requirements

**LoRA (Low-Rank Adaptation):**

- Add low-rank matrices: $W' = W + AB^T$ where $A, B$ are small matrices
- Freeze original weights, train only $A, B$
- Drastically fewer parameters
- Simple, effective adaptation

**QLoRA:**

- Quantize base model to int8
- Apply LoRA on top
- 4x memory savings
- Feasible on single GPU for large models

**Prompt Tuning:**

- Learn soft prompts (embeddings) prepended to input
- No weight modifications
- Minimal new parameters
- Useful for prompt engineering

- **Production Layer:**
  - Efficient fine-tuning for domain adaptation
  - Multiple LoRA adapters for different tasks
  - Merging LoRA weights for inference

---

#### **14.4 Inference Optimization & Serving**

- **Mathematical Layer & Implementation Layer:**

**KV Cache:**

- During autoregressive generation, previously computed key/value vectors are reusable
- Store them to avoid recomputation
- Significant speedup with minimal memory increase
- Trade-off: more memory during inference

**Speculative Decoding:**

- Use smaller model for draft generation
- Verify drafts with larger model
- Accept drafts or resample
- Faster overall generation

**Quantization:**

- int8, int4 quantization of weights
- Minimal accuracy loss
- Faster inference, smaller model
- Combinations with KV cache quantization

**Batching & Paging:**

- Batch multiple requests together
- vLLM: virtual token memory for efficient batching
- Paging for different sequence lengths in batch

- **Implementation Layer:**
  - Using HuggingFace transformers library
  - vLLM for high-throughput serving
  - TensorRT for NVIDIA optimization
  - Ollama for local LLM serving
- **Production Layer:**
  - Real-time inference API
  - Batch inference pipelines
  - Latency requirements and trade-offs

---

### Month 15: Prompt Engineering & In-Context Learning

#### **15.1 Prompt Engineering**

- **Implementation Layer:**
  - Zero-shot prompting
  - Few-shot prompting (providing examples)
  - Chain-of-thought prompting: reasoning step-by-step
  - Self-consistency: sampling multiple reasoning paths
  - Retrieval-augmented generation (RAG): providing context
  - Role-based prompting
- **Systems Layer:**
  - Prompt templates and reusability
  - Prompt versioning and A/B testing
- **Production Layer:**
  - Prompt optimization for downstream tasks
  - Evaluating prompt effectiveness

---

#### **15.2 RAG & Knowledge Augmentation**

- **Components:**
  - Retriever: find relevant context from knowledge base
  - LLM: generate response with context
  - Indexing: organizing knowledge efficiently
- **Implementation Layer:**
  - Vector similarity for retrieval (embeddings, cosine similarity)
  - Using LangChain or similar frameworks
  - Integration with vector databases (FAISS, Pinecone, Weaviate)
- **Production Layer:**
  - Building knowledge bases
  - Updating knowledge bases
  - Latency of retrieval + generation

---

### Month 16: Generative Models

#### **16.1 Autoencoders & VAE**

- **Mathematical Layer:**
  - Autoencoder: encoder $z = f(x)$, decoder $\hat{x} = g(z)$
  - Reconstruction loss: $L = \|x - \hat{x}\|^2$
  - Bottleneck for learning compressed representation
  - Variational Autoencoder (VAE): probabilistic interpretation
  - variational lower bound (ELBO): $\log p(x) \geq E_{q(z|x)}[\log p(x|z)] - KL(q(z|x) \| p(z))$
  - Reparameterization trick: $z = \mu + \sigma \cdot \epsilon$ where $\epsilon \sim \mathcal{N}(0, 1)$
- **Implementation Layer:**
  - Autoencoder architecture
  - VAE loss computation
  - PyTorch: encoder/decoder networks, loss functions
  - Sampling from latent space
- **Production Layer:**
  - Anomaly detection with reconstruction error
  - Data generation and interpolation

---

#### **16.2 Generative Adversarial Networks (GANs)**

- **Mathematical Layer:**
  - Generator: $G$ produces fake samples from random noise
  - Discriminator: $D$ classifies real vs fake
  - Minimax game: $\min_G \max_D E[log D(x)] + E[\log(1 - D(G(z)))]$
  - Mode collapse problem (generator ignores some data modes)
  - Training instability
  - Wasserstein GAN: using Wasserstein distance instead of JS divergence
- **Implementation Layer:**
  - Generator and Discriminator architectures
  - PyTorch GAN implementation
  - Different loss functions (standard, Wasserstein, hinge)
  - Spectral normalization for stability
- **Architectures:**
  - DCGAN: convolutional GAN architecture
  - StyleGAN: style-based generation with disentanglement
  - Conditional GAN: conditioned generation
- **Production Layer:**
  - Image synthesis at scale
  - Style transfer applications

---

#### **16.3 Diffusion Models**

- **Mathematical Layer:**
  - Forward process (diffusion): gradually add noise to data
  - Reverse process: learn to remove noise (denoising)
  - Noise schedule: $\sigma_t = \sqrt{1 - \bar{\alpha}_t}$ variance at step $t$
  - Denoising score matching: learning score function (gradient of log probability)
  - DDPM (Denoising Diffusion Probabilistic Models): Markov chain
  - DDIM (Denoising Diffusion Implicit Models): faster sampling with fewer steps
- **Implementation Layer:**
  - Forward diffusion process
  - Training denoising network (UNet architecture)
  - Sampling (ancestral, DDIM)
  - Classifier guidance for control
- **Architectures:**
  - Latent diffusion: diffusion in latent space (much more efficient)
  - Score-based generative models
- **Production Layer:**
  - Efficient sampling strategies
  - Conditioning and control
  - Text-to-image models (Stable Diffusion, DALL-E)

---

#### **16.4 Flow-based Models**

- **Mathematical Layer:**
  - Normalizing flows: invertible transformations
  - Change of variables formula for probability
  - Coupling layers for efficiency (alternating dimensions)
  - Autoregressive flows (sequential dependencies)
- **Implementation Layer:**
  - Implementing simple flows
  - Neural autoregressive flows
- **Production Layer:**
  - Exact likelihood computation (advantage over VAE/GAN)
  - Sampling efficiency

---

### Month 17: Reinforcement Learning

#### **17.1 Markov Decision Processes & Value-Based Methods**

- **Mathematical Layer:**
  - MDP: states, actions, transitions, rewards
  - Markov property: future independent of past given present
  - Value function $V(s) = E[R_t | s_t = s]$: expected cumulative reward
  - Action-value function $Q(s, a) = E[R_t | s_t = s, a_t = a]$
  - Bellman equation: $V(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')]$
  - Discount factor $\gamma$: present value of future rewards
  - Value iteration: iteratively updating value estimates
  - Policy iteration: evaluation (estimate V) then improvement (greedy policy)
- **Implementation Layer:**
  - Value iteration algorithm
  - Policy iteration algorithm
  - Grid world environments
  - OpenAI Gym for environments
- **Production Layer:**
  - Planning algorithms for known environments

---

#### **17.2 Deep Q-Learning & Policy Gradient Methods**

- **Mathematical Layer:**
  - Q-learning: off-policy value iteration
  - Experience replay: storing and reusing past transitions
  - Target network: separate network for stability
  - Double DQN: addressing overestimation in Q-learning
  - Dueling DQN: decomposing $Q(s,a) = V(s) + A(s,a) - \bar{A}(s)$
- **POLICY GRADIENT METHODS:**
  - Policy gradient: directly learning policy $\pi_\theta$
  - REINFORCE: $\nabla J = E[\nabla \log \pi_\theta(a|s) R_t]$
  - Advantage Actor-Critic: using critic $V(s)$ to reduce variance
  - Policy Proximal Policy Optimization (PPO): clipping for stability
  - Trust Region Policy Optimization (TRPO): trust region constraint
- **Implementation Layer:**
  - DQN network architecture
  - Experience replay buffer
  - PyTorch: policy networks, value networks
  - OpenAI Gym environment interaction
- **Production Layer:**
  - Simulation environments for RL
  - Real-world RL safety considerations

---

#### **17.3 Advanced RL & Imitation Learning**

- **Mathematical Layer & Implementation Layer:**
  - Model-based RL: learning environment model
  - Imagination-augmented agents
  - Imitation learning (learning from demonstrations)
  - Behavioral cloning vs inverse reinforcement learning
  - Learning from human feedback: preference learning
- **Production Layer:**
  - RL for game playing
  - Robotics applications
  - RLHF for LLM fine-tuning

---

### Month 18: Production ML Systems & Deployment

#### **18.1 Model Compression & Optimization**

- **Mathematical Layer:**

**Quantization:**

- Post-training quantization: convert float32 to int8 after training
- Quantization-aware training: simulate quantization during training
- Symmetric vs asymmetric quantization
- Per-channel vs per-layer quantization
- Non-uniform quantization (different levels for different value ranges)

**Pruning:**

- Structured pruning: remove entire channels/filters
- Unstructured pruning: remove individual weights
- Magnitude pruning: remove small weights
- Lottery ticket hypothesis: finding sparse subnetworks

**Knowledge Distillation:**

- Teacher model (large, accurate) teaches student model (small)
- Soft targets: using teacher softmax outputs with temperature
- Loss: discrepancy between student and teacher predictions
- Temperature scaling for softer targets

**Low-Rank Factorization:**

- Weight matrix $W \approx UV^T$ with $U, V$ low-rank
- Reduces parameters significantly
- Training with low-rank constraint

- **Implementation Layer:**
  - PyTorch quantization API
  - Pruning with magnitude criteria
  - Implementing knowledge distillation
- **Hardware Layer:**
  - Inference speedup from quantization (int8 operations faster)
  - Memory reduction
  - Hardware-specific optimizations (TensorRT, OpenVINO)
- **Production Layer:**
  - Compression for mobile/edge deployment
  - Post-quantization tuning for minimal accuracy loss
  - Accuracy-latency trade-offs

---

#### **18.2 Model Serving & APIs**

- **Implementation Layer:**
  - REST APIs with FastAPI/Flask
  - Request/response formats (JSON)
  - Batching requests for throughput
  - gRPC for lower latency
  - GraphQL for flexibility
- **Systems Layer:**
  - Model loading and caching
  - Concurrent request handling
  - Health checks and graceful shutdown
  - Versioning and A/B testing
- **Production Layer:**
  - Load balancing across model replicas
  - Horizontal scaling
  - Canary deployments
  - Monitoring and alerting

---

#### **18.3 Distributed Training**

- **Mathematical Layer:**
  - Data parallelism: same model, different data on different GPUs
  - Model parallelism: different model partitions on different GPUs
  - Pipeline parallelism: sequential stages on different GPUs
  - Gradients averaging (AllReduce operations)
- **Implementation Layer:**
  - PyTorch distributed training with `torch.nn.parallel.DistributedDataParallel`
  - HuggingFace accelerate library
  - Synchronization points (barrier operations)
  - Gradient synchronization
- **Hardware Layer:**
  - Inter-GPU communication: NVLINK, Infiniband
  - All-reduce algorithms (tree, ring topologies)
  - Communication overhead minimization
- **Production Layer:**
  - Fault tolerance and checkpointing
  - Scaling to hundreds of GPUs
  - Efficient resource utilization

---

#### **18.4 MLOps & Production Monitoring**

- **Systems Layer:**
  - Experiment tracking (MLflow, Weights & Biases)
  - Hyperparameter versioning and reproducibility
  - Dataset versioning (DVC, Git LFS)
  - Model registry and versioning
- **Monitoring:**
  - **Model performance monitoring:**
    - Prediction drift: model's predictions changing over time
    - Data drift: input distribution changes
    - Concept drift: relationship between features and labels changes
    - Ground truth drift: label distribution changes
  - **System monitoring:**
    - Latency and throughput
    - GPU/CPU utilization
    - Error rates
    - Request volume

- **Implementation Layer:**
  - Monitoring tools: Prometheus, Grafana
  - Data drift detection: statistical tests, distribution comparison
  - Model performance dashboards
  - Alert thresholds and automation
- **Production Layer:**
  - Automated retraining triggers
  - A/B testing framework
  - Rollback strategies
  - Incident response procedures

---

#### **18.5 Capstone Project 4: Production ML System**

**End-to-end ML system deployment:**

1. **Problem Selection:**
   - Real business problem
   - Data sources and pipelines
   - Success metrics

2. **Model Development:**
   - Data collection and preprocessing
   - Feature engineering
   - Model selection and training
   - Hyperparameter tuning
   - Evaluation and error analysis

3. **Optimization & Compression:**
   - Model profiling (latency, memory)
   - Quantization or pruning if needed
   - Benchmark on target hardware

4. **API Development:**
   - REST API with FastAPI
   - Input validation and error handling
   - Batching for throughput
   - Request logging

5. **Containerization:**
   - Dockerfile with minimal image
   - Multi-stage build
   - GPU support (nvidia-docker)
   - Docker Compose for local testing

6. **Deployment:**
   - Kubernetes manifests (Deployment, Service, etc.)
   - Health checks and resource limits
   - Horizontal Pod Autoscaling
   - Canary deployment strategy

7. **Monitoring & Maintenance:**
   - Performance metricsdashboard
   - Data and prediction drift detection
   - Automated alerting
   - Retraining pipeline
   - Model registry integration

8. **Documentation:**
   - Model card
   - API documentation
   - Deployment guide
   - Troubleshooting guide
   - Known limitations and biases

---

---

## **CAPSTONE PROJECTS TIMELINE**

### **Capstone 1 (Month 6): Classical ML Project**

**Duration:** 4 weeks

- End-to-end ML pipeline
- EDA, feature engineering, model selection
- Evaluation and error analysis
- Simple API and containerization
- **Deliverables:** Code, model, documentation, presentation

### **Capstone 2 (Month 12): Computer Vision Project**

**Duration:** 4 weeks

- CNN model (transfer learning)
- Object detection or segmentation
- Performance optimization
- REST API serving
- **Deliverables:** Code, trained model, deployment guide

### **Capstone 3 (Month 15): NLP Project**

**Duration:** 4 weeks

- Transformer model fine-tuning
- Classification or generation task
- Prompt engineering exploration
- API with RAG if applicable
- **Deliverables:** Code, fine-tuned model, API

### **Capstone 4 (Month 18): Complete Production System**

**Duration:** 4 weeks

- Full ML pipeline (data to serving)
- Distributed training or large-scale inference
- Kubernetes deployment
- Monitoring and retraining
- **Deliverables:** Docker images, K8s manifests, monitoring setup, documentation

---

## **LEARNING METHODOLOGY**

For each topic, provide:

1. **Intuitive Explanation:**
   - Why does this matter?
   - What problem does it solve?
   - Real-world analogies

2. **Detailed Mathematical Derivation:**
   - Starting from first principles
   - Step-by-step proofs
   - Assumptions and limitations

3. **Visual Diagrams:**
   - Architecture diagrams
   - Data flow
   - Computation graphs
   - Hardware execution

4. **Implementation from Scratch:**
   - Code without libraries first
   - Understanding every operation
   - Performance trade-offs

5. **Library Implementation:**
   - PyTorch, TensorFlow, scikit-learn
   - When and why to use each
   - API deep dives

6. **Hardware Implications:**
   - CPU vs GPU considerations
   - Memory hierarchies
   - Optimization opportunities
   - Profiling and bottleneck identification

7. **Production Considerations:**
   - Scalability
   - Latency and throughput
   - Cost efficiency
   - Monitoring and maintenance

8. **Real-World Examples:**
   - Published papers and implementations
   - Industry case studies
   - Open-source models and datasets

---

## **ASSESSMENT STRATEGY**

### **Continuous Assessment (Weekly):**

- **Theory Quizzes:** Core concepts from that week
- **Implementation Assignments:** Code from scratch + library usage
- **Code Reviews:** Peer feedback on implementation quality
- **Reflection Journals:** Learning process documentation

### **Monthly Projects:**

- **Month 1-3:** Implement foundational algorithms (matrix operations, gradient descent, backprop)
- **Month 4-9:** Classical ML end-to-end projects
- **Month 10-15:** Deep learning models and optimization
- **Month 16-18:** Production systems and deployment

### **Capstone Presentations:**

- **Month 6, 12, 15, 18:** Final project presentations to cohort
- Live Q&A on design decisions
- Code quality and documentation review
- Production readiness assessment

### **Final Evaluation:**

- **Portfolio Review:** Best 3-4 projects
- **Live Coding Interview:** Implement algorithm under time pressure
- **System Design:** Design production ML system from scratch
- **Oral Exam:** Theory and application across domains

---

## **RESOURCE INDEX**

### **Key Papers & References:**

**Foundations:**

- LeCun, Bengio, Hinton (2015): "Deep Learning" (Nature review)
- Goodfellow et al.: "Deep Learning" (textbook)

**Classical ML:**

- Hastie, Tibshirani, James: "Statistical Learning" (textbook)
- Scikit-learn documentation

**Deep Learning:**

- Krizhevsky et al. (2012): AlexNet (ImageNet breakthrough)
- He et al. (2016): ResNet (residual networks)
- Vaswani et al. (2017): "Attention is All You Need" (Transformers)

**LLMs:**

- Devlin et al. (2019): BERT
- Radford et al. (2019): Language Models are Unsupervised Multitask Learners (GPT-2)
- Brown et al. (2020): Language Models as Few-Shot Learners (GPT-3)
- OpenAI (2023): GPT-4 Technical Report

**Diffusion:**

- Ho et al. (2020): DDPM
- Song et al. (2020): Score-Based Generative Modeling

**Reinforcement Learning:**

- Sutton & Barto (2018): "Reinforcement Learning: An Introduction"
- Mnih et al. (2015): DQN

**Production ML:**

- Sculley et al.: "Machine Learning: The High-Interest Credit Card of Technical Debt"
- Polyzotis et al.: DataFlow for ML

### **Online Resources:**

- Papers With Code (papers + implementations)
- Hugging Face (transformers, datasets, models)
- Papers.cv (computer vision papers)
- Distill.pub (interactive machine learning explanations)
- Fast.ai (practical courses)
- NVIDIA Deep Learning Institute (GPU programming)

### **Datasets:**

- ImageNet (computer vision)
- COCO (object detection, segmentation)
- WikiText (language modeling)
- Common Crawl (web-scale text)
- OpenML (diverse ML datasets)

---

## **SUCCESS CRITERIA**

Students completing this 18-month program will be able to:

✅ **Theoretical:** Derive and understand algorithms from mathematical first principles

✅ **Implementation:** Code algorithms from scratch and use production libraries

✅ **Systems:** Optimize for hardware (CPU/GPU), deploy at scale, monitor in production

✅ **Practical:** Build end-to-end ML systems from problem definition to monitoring

✅ **Communication:** Explain complex concepts intuitively and present technical work clearly

✅ **Research:** Implement state-of-the-art models from papers, understand novel architectures

✅ **Production:** Deploy models in containers, serve APIs, implement MLOps practices

✅ **Adaptability:** Learn new architectures and techniques as field evolves

---

**Final Note:**

This is a demanding curriculum designed for highly committed students. The combination of deep theory, careful implementation, and production focus ensures graduates are not just capable of implementing existing techniques but truly understand the foundational principles. Success requires consistency, curiosity, and a willingness to struggle with difficult concepts.

The 18-month timeline is ambitious but achievable for full-time students. Part-time adaptations should expand to 36 months while maintaining the same depth.

---

_Last Updated: March 2026_
_Version: 1.0_
