# 📘 Exploratory Data Analysis & Statistical Testing Notebook

This repository contains an end-to-end notebook for exploring data distributions, checking for normality, and applying the right statistical tests before using the data in machine learning models.

---

## 🎯 Project Overview

The notebook demonstrates:
- How to inspect and visualize data distributions  
- How to measure **skewness** and **kurtosis**  
- How to test for **normality** using statistical tests  
- How to apply **data transformations** to improve normality  
- When to use **parametric** vs **non-parametric** tests  
- Why all this matters in **machine learning preprocessing**

---

## 🧩 Workflow Summary

### 1. Importing Libraries
Uses:
- **NumPy**, **Pandas** for data handling  
- **Matplotlib**, **Seaborn** for visualization  
- **SciPy.stats** for statistical tests  
- **Scikit-learn** for transformations and scaling  

---

### 2. Reading & Inspecting Data
The dataset is loaded (e.g., `Mall_Customers.csv`) and inspected for:
- Columns, data types, and shape  
- Missing values and unique counts  
- Basic descriptive statistics  

---

### 3. Visualizing the Data
Histograms and distribution plots are used to understand how features like:
- **Age**, **Annual Income**, and **Spending Score**  
are distributed, including comparisons between genders.

---

### 4. Understanding Skewness & Kurtosis

#### Skewness
Explains whether data is:
- **Right-skewed** (long right tail)  
- **Left-skewed** (long left tail)  
- **Symmetric**

| Skew Type | Meaning | Fix |
|------------|----------|-----|
| Right-skewed | Many small values, few large ones | `log(x+1)`, `sqrt(x)`, Box–Cox |
| Left-skewed | Many large values, few small ones | `x²`, `exp(x)` |
| No skew | Balanced | No action needed |

#### Kurtosis
Indicates tail heaviness and peak sharpness.

| Kurtosis | Shape | Implication |
|-----------|--------|-------------|
| < 0 | Platykurtic (flat) | Evenly spread, fewer outliers |
| ≈ 0 | Mesokurtic (normal) | Typical distribution |
| > 0 | Leptokurtic (sharp) | More outliers, heavy tails |

---

### 5. Transformations for Normality
Several transformations were applied and visually compared:

| Method | Works For | Example |
|---------|------------|----------|
| Log Transform | Right-skewed data | `np.log1p(x)` |
| Square Root | Mild skew | `np.sqrt(x)` |
| Box–Cox | Positive data only | `scipy.stats.boxcox(x)` |
| Yeo–Johnson | Any data (pos/neg) | `PowerTransformer(method='yeo-johnson')` |

Distribution plots before and after transformation show improved symmetry and bell-shape.

---

### 6. Normality Tests

Four normality tests were conducted to statistically assess whether data follows a normal distribution.

| Test | Purpose | Key Idea |
|------|----------|-----------|
| **Shapiro–Wilk** | Tests for normality | Works best for small samples |
| **Kolmogorov–Smirnov (KS)** | Compares data vs. reference | Measures goodness-of-fit |
| **Anderson–Darling** | Tail-sensitive test | Compares against critical values |
| **D’Agostino–Pearson K²** | Uses skewness & kurtosis | Detects subtle deviations |

#### Example Results
- **Shapiro–Wilk p < 0.05** → Not normal  
- **KS p < 0.05** → Not normal  
- **Anderson–Darling statistic > critical values** → Not normal  
- **D’Agostino p < 0.05** → Not normal  

🧠 *Conclusion:* The **Age** variable deviates from normal distribution despite mild skew/kurtosis — large samples make tests more sensitive.

---

### 7. Interpreting Normality Tests

All tests check the same null hypothesis:

> **H₀:** Data is normally distributed  
> **H₁:** Data is not normally distributed

| Result | Interpretation |
|---------|----------------|
| p > 0.05 | ✅ Data likely normal |
| p ≤ 0.05 | ❌ Data not normal |

For Anderson–Darling:  
If statistic < critical value → normal; else → not normal.

---

### 8. Impact on Machine Learning Models

| Model Type | Normality Needed? | Notes |
|-------------|------------------|-------|
| Linear Regression, Logistic Regression | ✅ Yes | Assumes normally distributed errors |
| PCA, LDA | ✅ Yes | Normality improves projection |
| Tree-based models (RF, XGBoost) | ❌ No | Split by thresholds |
| Neural Networks | ⚙️ Partially | Scaling helps convergence |
| KNN, SVM | ⚙️ Sometimes | Sensitive to scale/distribution |

Failing a normality test doesn’t “break” ML — it only means you may need transformations or robust models.

---

### 9. Parametric vs Non-Parametric Tests

#### Parametric Tests (assume normality)
- **t-test** — compare two means  
- **ANOVA** — compare three or more means  
- **Pearson correlation**

#### Non-Parametric Tests (no normality assumption)
Used when normality tests fail.  
Examples:
- **Mann–Whitney U**  
- **Kruskal–Wallis**  
- **Spearman correlation**

They work on **ranks** or **medians**, not means.

---

### 10. Key Insights & Takeaways

- Most real-world data is *not perfectly normal* — and that’s fine.  
- Use visual + statistical checks together (histograms, Q–Q plots).  
- For large datasets, rely more on visual judgment than strict p-values.  
- **Scaling ≠ Normalizing:**
  - Scaling changes range/variance.  
  - Normalizing changes distribution shape.  

---

### 11. Scaling Summary

| Scaler | Output Range | Outlier Handling | Notes |
|---------|---------------|------------------|-------|
| MinMaxScaler | [0,1] | No | Sensitive to outliers |
| StandardScaler | Any | No | Mean = 0, Std = 1 |
| RobustScaler | Any | Yes | Uses median & IQR |
| QuantileTransformer | [0,1] or normal | Yes | Maps to uniform/normal |
| PowerTransformer | Any | Some | Reduces skew |

---

### 12. Conclusion

This notebook provides a complete workflow for:
1. Exploring and visualizing data  
2. Measuring skewness and kurtosis  
3. Testing for normality  
4. Applying appropriate transformations  
5. Choosing correct parametric or non-parametric tests  
6. Understanding how data shape affects machine learning


## How to Run

Follow these steps to set up and run the project:

1. **Clone the repository**  
git clone <your-repo-url>  
cd <your-repo-folder>

2. **Install dependencies**  
Make sure Python and pip are installed on your system, then run:  
pip install -r requirements.txt

3. **Open the notebook**  
You can open `main.ipynb` using either:  
- VS Code Jupyter extension: Open VS Code and open `main.ipynb`.  
- Jupyter Notebook: This will open a browser window where you can select `main.ipynb`.

4. **Run the notebook cells**  
Click the **Run ▶️** button on each cell or press `Shift + Enter`.  
The notebook will automatically use the CSV file located in the `data/` folder.

5. **Verify the CSV file**  
Ensure that your data file exists at:  
data/your_data.csv  
The notebook reads data from this file, so it must be present to run correctly.

## Author and Contact

**Author:** Mohammad Hammad Ahmad 
**Email:** mdhammadahmadgithub@gmail.com 
**LinkedIn:** www.linkedin.com/in/mohammad-hammad-ahmad-188628227  

Feel free to reach out via email or on LinkedIn.



