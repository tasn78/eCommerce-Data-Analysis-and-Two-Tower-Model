# eCommerce-Data-Analysis-and-Two-Tower-Model
Analysis of an eCommerce dataset from Kaggle. Preprocessing, data analysis and multiple recommendation models with a final Two Tower Architecture for recommendations

While the .py was provided, all work done on this project was completed in Google Colab.

Dataset - eCommerce behavior data from multi category store
https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
Size: ~7M rows across two months (2019-Oct.csv, 2019-Nov.csv)
Columns: event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session.

This project analyzes user behavior on a large-scale e-commerce dataset (Oct–Nov 2019) and builds a recommendation system powered by Neural Collaborative Filtering (NCF).
The pipeline includes data preprocessing, exploratory data analysis (EDA), user/product behavior insights, association rule mining (Apriori), and a production-style embedding-based recommender model.

├── data_raw/               # Original CSV files
├── data_clean/             # Cleaned and preprocessed datasets
├── notebooks/              # Exploratory analysis & visualization
├── src/
│   ├── preprocessing.py    # Cleaning & filling missing values
│   ├── analysis.py         # EDA and visualization scripts
│   ├── association_rules.py# Product/category affinity mining
│   ├── ncf_model.py        # Neural Collaborative Filtering implementation
│   └── recommend.py        # Recommendation inference function
├── results/
│   ├── plots/              # Generated plots (price, time analysis, etc.)
│   └── rules/              # Saved association rule CSVs
└── README.md               # This file

Main dependencies:
Python 3.10+
pandas, numpy, matplotlib, seaborn
scikit-learn
mlxtend (Apriori, association rules)
tensorflow / keras
faiss-cpu (optional, for scalable similarity search)

Data Preprocessing
Brand & category fill: Used category_id → brand mappings to impute missing brand values.
Unknown handling: Filled null brand and category_code with unknown_brand / unknown_category.
Filtering: Dropped rows missing user_session to ensure event-level integrity.
Merging: Combined October and November datasets into a single clean artifact.

Outputs:
clean_copy_oct.csv, clean_copy_nov.csv, clean_filled_oct.csv, clean_filled_nov.csv

Exploratory Data Analysis (EDA)
Key analyses performed:
Event distribution: View → Cart → Purchase funnel across months.
Temporal patterns: Purchase activity by hour, day of week, and month.
Category & brand trends: Top categories/brands, price distribution.
User/session metrics: Purchase frequency per user, events per session.
Affinity mining: Apriori algorithm to find frequent itemsets & association rules.
Network visualization: Product affinity network graph (top 50 rules).


Recommendation System
We implemented a Neural Collaborative Filtering (NCF) model:
Embeddings: Learned 32-dimensional user & product embeddings.
Negative Sampling: Generated synthetic negative samples to balance training.
Training: Adam optimizer, early stopping, 80/20 train-test split.
Post-processing: Diversified recommendations using category balancing.

Two-Tower Architecture
Our recommendation engine uses a **two-tower neural network architecture**, where:
- **User Tower:** Learns a dense embedding representation for each user.
- **Item Tower:** Learns a dense embedding representation for each product.
- **Interaction Layer:** Combines the two embeddings via concatenation and passes them through fully connected layers to predict purchase likelihood.
- **Training Objective:** Binary cross-entropy with negative sampling to model implicit feedback.

This approach scales well for large e-commerce datasets and allows for:
- **Embedding-based retrieval:** Users and items are represented in the same vector space, enabling fast top-N recommendations.
- **Cold-start extension:** Can be extended by enriching towers with side features (e.g., user demographics, item categories).
- **Diversification:** Our final recommendations include category-aware diversity re-ranking to improve coverage.

### Example Usage
python
from recommend import recommend_products_for_user
recommendations = recommend_products_for_user(user_id=123456, top_n=5, diversity_weight=0.3)
print(recommendations)

This architecture is modular — the learned embeddings can be exported and indexed using FAISS for approximate nearest neighbor search to serve real-time recommendations at scale.

Results
Top products & categories: Smartphones consistently dominated views and purchases.
Conversion rate insight: Higher conversion in October → potential seasonal dip in Nov.
NCF performance: Achieved meaningful user–item prediction accuracy with efficient sampling (10% users, top 10k products).
Association rules: Generated >500 frequent itemsets, enabling product bundling insights.

Future Work
Integrate FAISS for large-scale nearest neighbor retrieval.
Add context-aware features (time, session length).
Deploy as a REST API (Flask/FastAPI) for real-time recommendations.
Experiment with two-tower models for better generalization.

Authors
Tom S – Data Preprocessing, EDA, NCF Implementation
Collaborators: [Amy], [Vy], [Priya] – Cleaning, Null Handling, and Exploratory Analysis.
