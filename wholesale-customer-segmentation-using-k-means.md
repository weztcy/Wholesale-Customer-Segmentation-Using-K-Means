# Portfolio Project Data — Wholesale Customer Segmentation Using K-Means

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/wholesale_customer_segmentation_using/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image11.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "wholesale-customer-segmentation-using-k-means",
  "code": "PRO-24 // ML-09",
  "title": "Wholesale Customer Segmentation Using K-Means",
  "category": "machine-learning",
  "categoryLabel": "Unsupervised Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Wholesale-Customer-Segmentation-Using-K-Means",
  "license": "MIT License",
  "banner": "/images/projects/wholesale_customer_segmentation_using/banner.webp",
  "summary": "Unsupervised K-Means segmentation of 440 wholesale clients using log1p scaling, multi-model benchmarking, PCA projection, and Streamlit analytics.",
  "projectDescription": "An end-to-end unsupervised learning pipeline that segments 440 wholesale distributor accounts across six annual spending categories. Using log1p skewness compression, StandardScaler, multi-algorithm benchmarking, and 2D PCA projection, it uncovers three actionable buyer personas served in Streamlit.",
  "architectureSummary": "Raw CSV Audit -> Log1p + StandardScaler -> Multi-Model Sweep (k=2..10) -> K-Means (k=3) & 2D PCA -> CSV Profiles & Streamlit Dashboard",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "Pandas",
    "NumPy",
    "SciPy",
    "Matplotlib",
    "Seaborn",
    "Streamlit",
    "Jupyter"
  ],
  "telemetryMatrix": {
    "projectType": "Unsupervised Clustering Pipeline",
    "targetPlatform": "Python & Streamlit Web Dashboard",
    "primaryDomain": "Wholesale B2B Customer Analytics",
    "architecturePipeline": "Log1p + StandardScaler -> K-Means + PCA",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Customer Segmentation Pipeline",
    "learningParadigm": "Unsupervised Learning",
    "modelDomain": "Wholesale B2B Spending Analytics",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Log1p Skew Normalization",
      "K-Means Clustering",
      "PCA Cluster Projection"
    ]
  },
  "mlProblemContext": "Wholesale distributors serve diverse Horeca and Retail accounts without ground-truth behavioral segment labels. Raw annual expenditure across six product categories exhibits severe right-skewness (+2.56 to +11.15) that distorts unscaled Euclidean centroids into isolating a few mega-spenders instead of meaningful customer tiers.",
  "mlDatasetOverview": {
    "profileBadge": "UCI Wholesale Dataset",
    "source": "UCI Machine Learning Repository (ID 292, Wholesale_Customers.csv)",
    "modality": "Tabular",
    "scale": "440 Client Records (14,619,500 m.u. Total Spend)",
    "inputShape": "440 x 6 Numerical Spending Matrix",
    "dataFormat": "CSV",
    "description": "Contains annual wholesale spending in monetary units across Fresh, Milk, Grocery, Frozen, Detergents_Paper, and Delicassen for 440 Portuguese clients, plus Channel and Region fields reserved exclusively for post-clustering profiling without train/test splitting.",
    "provenanceNote": "Collected by Margarida Cardoso (2013, DOI: 10.24432/C5030X) with zero missing values, zero duplicates, and positive minimum spending (>= 3 m.u.)."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "6D Spending Feature Space",
    "spaceNote": "Channel and Region are excluded from clustering inputs to prevent administrative label leakage.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Discrete Behavioral Cluster Assignment",
      "type": "Multi-Cluster Partition (k = 3)",
      "distribution": "Cluster 0: 145 (32.95%), Cluster 1: 212 (48.18%), Cluster 2: 83 (18.86%)"
    },
    "inputSignals": [
      {
        "name": "Fresh",
        "type": "Numerical (m.u.)",
        "importanceOrRole": "PC2 Driver (+0.590 loading)",
        "description": "Annual spending on fresh produce; highest nominal mean (12,000.30 m.u.) and primary separator for Horeca kitchen accounts."
      },
      {
        "name": "Grocery",
        "type": "Numerical (m.u.)",
        "importanceOrRole": "PC1 Driver (+0.571 loading)",
        "description": "Annual spending on staple pantry goods; strongly correlated with Detergents_Paper (r = 0.92 raw, 0.80 log)."
      },
      {
        "name": "Detergents_Paper",
        "type": "Numerical (m.u.)",
        "importanceOrRole": "PC1 Driver (+0.551 loading)",
        "description": "Annual spending on cleaning and paper goods; sharply distinguishes Retail Specialists (5,446 m.u.) from Horeca Focused buyers (426 m.u.)."
      },
      {
        "name": "Milk",
        "type": "Numerical (m.u.)",
        "importanceOrRole": "PC1 Driver (+0.542 loading)",
        "description": "Annual spending on dairy products; co-varies closely with Grocery (r = 0.73 raw) and anchors broad high-volume accounts."
      },
      {
        "name": "Frozen",
        "type": "Numerical (m.u.)",
        "importanceOrRole": "PC2 Driver (+0.590 loading)",
        "description": "Annual spending on frozen foods; elevated in Cluster 0 and Cluster 1 while minimal in shelf-stable Cluster 2 (608 m.u.)."
      },
      {
        "name": "Delicassen",
        "type": "Numerical (m.u.)",
        "importanceOrRole": "PC2 Driver (+0.530 loading)",
        "description": "Annual spending on specialty deli products; exhibits highest raw skewness (+11.15) compressed to -1.09 via log1p."
      }
    ],
    "dataCharacteristicsSummary": "All six spending variables display heavy right tails (skewness +2.56 to +11.15) and 4.55% to 9.77% IQR outliers representing legitimate bulk buyers. Log1p transformation compresses skewness to [-1.58, -0.22] and reveals strong Grocery-Detergents (r = 0.80) and Fresh-Frozen (r = 0.39) axes."
  },
  "mlLearningApproach": {
    "paradigm": "Unsupervised Learning (Centroid-Based Behavioral Clustering)",
    "methodologyDescription": "Three preprocessing pipelines (Raw+StandardScaler, Log1p+StandardScaler, Log1p+RobustScaler) are evaluated across K-Means, Agglomerative Ward, and Gaussian Mixture Models for k = 2..10. Multi-metric evaluation and business interpretability select K-Means (k = 3) on log-standardized inputs.",
    "justification": "Unlike k = 2, which merely mirrors the binary Horeca/Retail channel split, K-Means at k = 3 improves Davies-Bouldin Index (1.3482 vs 1.3515) and isolates Broad-Category Key Accounts generating 52.84% of revenue from pure pantry-focused retail buyers."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Data Quality & Schema Audit",
      "description": "Validated all 440 rows and 8 columns in Wholesale_Customers.csv, confirming zero missing values, zero duplicates, zero negative values, and positive minimums (>= 3 m.u.) while resolving metadata feature-count discrepancies.",
      "badge": "Quality Audit"
    },
    {
      "step": "02",
      "title": "Administrative Feature Exclusion",
      "description": "Excluded categorical Channel and Region columns from the modeling matrix so partitions reflect pure spending behavior rather than predefined sales territories.",
      "badge": "Leakage Guard"
    },
    {
      "step": "03",
      "title": "Log1p Skewness Compression",
      "description": "Retained high-volume bulk buyers as valid institutional accounts and applied np.log1p(x) across all six spending features, reducing skewness from [+2.56, +11.15] to [-1.58, -0.22].",
      "badge": "Log1p Transform"
    },
    {
      "step": "04",
      "title": "StandardScaler & 2D PCA Projection",
      "description": "Standardized log-transformed features to zero mean and unit variance for balanced Euclidean distance computation, then fit 2-component PCA capturing 71.27% cumulative variance.",
      "badge": "Feature Scaling"
    }
  ],
  "mlDataPreparationNote": "Channel and Region attributes are strictly withheld from scaling, clustering, and PCA fitting, entering only during post-clustering cross-tabulation.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "K-Means Clustering (k-means++ with Log1p + StandardScaler)",
    "architectureSummary": "Lloyd's centroid-partitioning algorithm initialized with k-means++ over 6D standardized log-spending space, paired with a 2D orthogonal PCA projection (PC1: 44.08% retail/pantry axis, PC2: 27.19% perishable/Horeca axis).",
    "configurationBadge": "k = 3 // n_init = 25",
    "hyperparameters": [
      {
        "name": "n_clusters",
        "value": "3",
        "description": "Defines three balanced behavioral segments: High Spenders, Horeca Focused, and Retail Specialists."
      },
      {
        "name": "init",
        "value": "k-means++",
        "description": "Probabilistic centroid seeding that accelerates convergence and prevents poor local minima."
      },
      {
        "name": "n_init",
        "value": "25",
        "description": "Executes 25 independent centroid restarts and retains the configuration with minimum WCSS inertia."
      },
      {
        "name": "random_state",
        "value": "42",
        "description": "Guarantees deterministic centroid initialization and reproducible PCA/GMM benchmark runs."
      }
    ],
    "objectiveCriterion": "Within-Cluster Sum of Squares (WCSS Inertia)",
    "optimizationSolver": "Lloyd's Iterative Expectation-Maximization with k-means++ Seeding",
    "trainingProcess": "All 81 benchmark configurations (3 pipelines x 3 algorithms x k = 2..10) are fitted systematically to expose raw-scale outlier traps. The final K-Means (k = 3) model converges on the 440 x 6 Log1p+StandardScaler matrix across 25 seeded initializations to achieve WCSS inertia of 1,553.41."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Multi-Metric Internal Cluster Validation & Pipeline Sensitivity Sweep (k = 2..10)",
    "description": "Evaluates geometric cohesion, inter-cluster separation, and segment balance across K-Means, Agglomerative Ward, and GMM over three scaling pipelines. Cluster-size auditing prevents inflated raw-scale Silhouette scores caused by 2-to-6-client outlier isolation.",
    "metrics": [
      {
        "label": "SILHOUETTE SCORE",
        "value": "0.2583",
        "benchmark": "Target > 0.25"
      },
      {
        "label": "DAVIES-BOULDIN",
        "value": "1.3482",
        "benchmark": "Baseline < 1.3515"
      },
      {
        "label": "CALINSKI-HARABASZ",
        "value": "152.84",
        "benchmark": "Target > 140.0"
      },
      {
        "label": "WCSS INERTIA",
        "value": "1,553.41",
        "benchmark": "Baseline < 1844.06"
      },
      {
        "label": "PCA VARIANCE",
        "value": "71.27%",
        "benchmark": "Target > 70.0%"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Dashboard — KPI Summary & 2D PCA Space",
      "badge": "Interactive UI",
      "imageUrl": "/images/projects/wholesale_customer_segmentation_using/image01.webp",
      "telemetryNote": "Displays global segment/channel/region filters, 440-client KPIs (14.62M m.u. total spend), PCA biplot, and segment counts."
    },
    {
      "title": "Streamlit Dashboard — Category Spend & Channel/Region Share",
      "badge": "Interactive UI",
      "imageUrl": "/images/projects/wholesale_customer_segmentation_using/image02.webp",
      "telemetryNote": "Compares filtered Mean vs Median annual spending alongside Channel (67.7% Horeca) and Region (71.8% Other) distributions."
    },
    {
      "title": "Streamlit Dashboard — Client Explorer & Strategic Playbook",
      "badge": "Interactive UI",
      "imageUrl": "/images/projects/wholesale_customer_segmentation_using/image03.webp",
      "telemetryNote": "Provides searchable client-level records with PCA coordinates and tailored retention, cold-chain, and pallet-pricing actions."
    },
    {
      "title": "Multi-Metric Model Comparison, Elbow & Silhouette Curves",
      "badge": "Model Selection",
      "imageUrl": "/images/projects/wholesale_customer_segmentation_using/image04.webp",
      "telemetryNote": "Benchmarks K-Means, Agglomerative Ward, and GMM across k = 2..10 and three scalers, confirming k = 3 stability."
    },
    {
      "title": "Raw Annual Spending Histograms & Skewness Profile",
      "badge": "EDA Distribution",
      "imageUrl": "/images/projects/wholesale_customer_segmentation_using/image05.webp",
      "telemetryNote": "Reveals severe right-skewness (+2.56 to +11.15) across all six spending categories prior to log1p normalization."
    },
    {
      "title": "2D PCA Projection Biplot of Customer Clusters",
      "badge": "PCA Biplot",
      "imageUrl": "/images/projects/wholesale_customer_segmentation_using/image06.webp",
      "telemetryNote": "Captures 71.27% variance across PC1 (44.08% pantry goods) and PC2 (27.19% perishables) with clear centroid separation."
    },
    {
      "title": "Spending Boxplots Before vs After Log1p Transformation",
      "badge": "Preprocessing",
      "imageUrl": "/images/projects/wholesale_customer_segmentation_using/image07.webp",
      "telemetryNote": "Demonstrates stabilization of >100k m.u. outliers into balanced log-scale interquartile ranges without dropping clients."
    },
    {
      "title": "Channel & Geographic Region Composition per Cluster",
      "badge": "Segment Profiling",
      "imageUrl": "/images/projects/wholesale_customer_segmentation_using/image08.webp",
      "telemetryNote": "Shows 98.6% Horeca concentration in Cluster 1 vs 61.4% and 60.2% Retail shares in Clusters 0 and 2."
    },
    {
      "title": "Raw-Scale Mean vs Median Spending Profiles by Cluster",
      "badge": "Spending Profile",
      "imageUrl": "/images/projects/wholesale_customer_segmentation_using/image09.webp",
      "telemetryNote": "Confirms Cluster 0 dominance across all categories (53,273 m.u. mean) vs Fresh/Frozen focus in Cluster 1 and Grocery focus in Cluster 2."
    },
    {
      "title": "Cluster Z-Score Heatmap & Population Distribution",
      "badge": "Cluster Heatmap",
      "imageUrl": "/images/projects/wholesale_customer_segmentation_using/image10.webp",
      "telemetryNote": "Contrasts all-positive Z-scores in Cluster 0 (+0.31 to +0.81) against sharp negative deviations in Clusters 1 and 2."
    },
    {
      "title": "Raw vs Log-Transformed Category Correlation Matrices",
      "badge": "Correlation",
      "imageUrl": "/images/projects/wholesale_customer_segmentation_using/image11.webp",
      "telemetryNote": "Highlights strong Grocery-Detergents_Paper correlation (r = 0.92 raw / 0.80 log) driving the PC1 retail axis."
    }
  ],
  "mlVisualizationTelemetryNote": "All 11 diagnostic and dashboard visuals correspond deterministically to the k = 3 Log1p + StandardScaler K-Means run seeded at 42.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Cluster separation is governed by two orthogonal purchasing axes: pantry/sanitation staples (Grocery, Detergents_Paper, Milk on PC1) and perishable kitchen inputs (Fresh, Frozen, Delicassen on PC2). Cluster 0 scores positively on both axes, whereas Clusters 1 and 2 specialize inversely.",
    "keyDrivers": [
      {
        "factor": "Grocery & Detergents_Paper Co-Spending",
        "impact": "Strongest linear pair (r = 0.92 raw, 0.80 log) anchoring PC1 (+0.571 / +0.551) and separating Cluster 2 Retail Specialists from Cluster 1."
      },
      {
        "factor": "Fresh & Frozen Perishable Demand",
        "impact": "Primary PC2 drivers (+0.590 each) distinguishing Cluster 1 Horeca accounts (+0.17 / +0.23 Z-score) from Cluster 2 (-1.16 / -1.12 Z-score)."
      },
      {
        "factor": "Multi-Category High-Volume Scale",
        "impact": "Cluster 0 clients (32.95%) maintain positive Z-scores across all 6 categories (+0.31 to +0.81), generating 52.84% (7.72M m.u.) of distributor revenue."
      }
    ],
    "limitations": [
      {
        "title": "Absence of Temporal & RFM Granularity",
        "description": "Data aggregates annual spend without transaction timestamps, preventing seasonality, recency, or order-frequency modeling."
      },
      {
        "title": "Revenue Without Margin & Cost Data",
        "description": "Spending is recorded in gross monetary units rather than net profit margin, so high-volume Cluster 0 discounts may alter true profitability."
      },
      {
        "title": "Historical Sample Size & Boundary Overlap",
        "description": "Covers 440 clients from 2013 Portugal with a 0.2583 Silhouette Score reflecting natural overlap between mid-sized retail and hospitality buyers."
      }
    ],
    "futureImprovements": [
      {
        "title": "RFM & Temporal Cohort Integration",
        "target": "Incorporate order-level timestamps to track quarterly seasonality and churn drift"
      },
      {
        "title": "Gross-Margin Weighted Segmentation",
        "target": "Integrate product-level margin and cold-chain delivery costs for profit-tiered pricing"
      },
      {
        "title": "Serialized Inference & Drift Monitoring",
        "target": "Export versioned scaler/K-Means pipelines via joblib for automated new-account scoring"
      }
    ],
    "discoveryNote": "Channel and Region correlations are descriptive post-hoc associations rather than causal rules; 38.62% of Key Accounts in Cluster 0 operate in Horeca."
  },
  "mlModelArtifacts": [
    {
      "name": "clustered_customers.csv",
      "format": "CSV"
    },
    {
      "name": "cluster_profiles.csv",
      "format": "CSV"
    },
    {
      "name": "model_comparison.csv",
      "format": "CSV"
    },
    {
      "name": "figures/*.png (11 Diagnostic Plots)",
      "format": "PNG"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Unsupervised Learning",
    "modelArchitecture": "K-Means with PCA",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "Multi-Metric Cluster Validation"
  }
}
```
