# 🩺 Cancer Prediction using Machine Learning

This project builds a **Machine Learning model** to predict whether a patient has **cancer** or not, based on cell sample data.  
It uses **Logistic Regression**, which is a popular **classification algorithm**.

---

## 📂 Project Files
- `Cancer_Prediction.ipynb` → Full Jupyter Notebook (explanations + code)  
- `Cancer_Prediction_CodeOnly.ipynb` → Notebook with only **code cells**  
- `Cancer_Prediction_CodeOnly.py` → Python script version (directly runnable)  
- `README.md` → Project documentation  

---

## 📊 Dataset
- Source: [YBI Foundation GitHub](https://github.com/YBIFoundation/Dataset/blob/main/Cancer.csv)  
- Data contains patient cell information like clump thickness, uniformity of cell size/shape, etc.  
- Target Column → `Class`  
  - `0` = No Cancer  
  - `1` = Cancer Detected  

---

## ⚙️ Steps in the Project
1. **Import Libraries**  
   - `pandas`, `numpy`, `sklearn`  

2. **Load Dataset**  
   ```python
   import pandas as pd
   cancer = pd.read_csv("https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv")

3. **Explore Data**

   * `cancer.head()`, `cancer.info()`, `cancer.describe()`

4. **Define Features & Target**

   ```python
   X = cancer.drop('Class', axis=1)
   y = cancer['Class']
   ```

5. **Split Dataset**

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

6. **Model Training**

   ```python
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression(max_iter=1000)
   model.fit(X_train, y_train)
   ```

7. **Prediction & Evaluation**

   ```python
   from sklearn.metrics import accuracy_score
   y_pred = model.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, y_pred))
   ```

---

## 🚀 How to Run

1. Install requirements:

   ```bash
   pip install pandas numpy scikit-learn
   ```
2. Run the Python script:

   ```bash
   python Cancer_Prediction_CodeOnly.py
   ```
3. Or open the Jupyter Notebook:

   ```bash
   jupyter notebook Cancer_Prediction.ipynb
   ```

---

## 🧠 Algorithm Explanation

* **Logistic Regression**

  * A supervised learning algorithm used for **binary classification**.
  * It calculates probability using the **sigmoid function**:

    $$
    P(y=1|X) = \frac{1}{1+e^{-(b_0 + b_1X_1 + b_2X_2 + ... + b_nX_n)}}
    $$
  * If probability > 0.5 → Predict **Cancer (1)**
  * Else → Predict **No Cancer (0)**

---

## 📊 Results

* **Accuracy:** \~95% (on test dataset)
* **Example Output:**

  ```
  Accuracy: 0.9523809523809523
  ```
* **Sample Prediction:**

  ```python
  sample = X_test.iloc[0].values.reshape(1, -1)
  print("Predicted:", model.predict(sample))
  print("Actual:", y_test.iloc[0])
  ```

  Output:

  ```
  Predicted: [1]
  Actual: 1
  ```

---

## 📷 Screenshots (Add later)

* Dataset Preview
* Accuracy Score Output
* Prediction Example

---

## 👨‍💻 Author

* Project Source: **YBI Foundation Bootcamp**
