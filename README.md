# 🚗 Car Price Prediction

A machine learning project that predicts the **selling price of a used car (in ₹)** from its age, kilometres driven, mileage, engine size, power, seats, seller type, fuel type and transmission. The best model (Random Forest) is served through an interactive **Streamlit** web app.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![MySQL](https://img.shields.io/badge/MySQL-Database-lightgrey)

---

## 📸 Screenshots

### 1. Input form
![Input form]<img width="1366" height="734" alt="Screenshot (2063)" src="https://github.com/user-attachments/assets/ce9f90cb-1b46-42f7-8775-d4f4aca425d0" />


### 2. Vehicle details
![Vehicle details]<img width="1366" height="734" alt="Screenshot (2064)" src="https://github.com/user-attachments/assets/068c93ff-d47a-4a3d-9889-41293433c67b" />


### 3. Predicted price
![Predicted price]<img width="1366" height="721" alt="Screenshot (2068)" src="https://github.com/user-attachments/assets/515e9035-c103-468d-add8-a4ebb12cabd1" />


---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Tech Stack](#-tech-stack)
3. [Project Structure](#-project-structure)
4. [Dataset](#-dataset)
5. [Project Workflow](#-project-workflow)
6. [Model Results](#-model-results)
7. [Installation and Setup (Terminal Commands)](#-installation-and-setup-terminal-commands)
8. [Run the Streamlit App](#-run-the-streamlit-app)
9. [How to Use the App](#-how-to-use-the-app)
10. [Model Input Features](#-model-input-features)
11. [Troubleshooting](#-troubleshooting)
12. [Future Improvements](#-future-improvements)
13. [Author](#-author)

---

## 📖 Project Overview

Buying or selling a used car and not knowing a fair price is a common problem. This project builds a regression model on a used-car dataset and wraps it in a simple web app, so anyone can enter car details and instantly get an estimated price.

**What the project covers:**

- Data collection, cleaning and validation
- Storing the cleaned data in a **MySQL** database and reading it back with pandas
- Exploratory Data Analysis (univariate, bivariate and multivariate)
- Feature engineering (one-hot encoding, feature scaling)
- Training and comparing **6 regression models**
- Hyperparameter tuning with **GridSearchCV**
- Saving the trained model and scaler with `pickle`
- A **Streamlit** web app for live predictions

---

## 🛠 Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.10 |
| Data handling | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | scikit-learn |
| Database | MySQL, SQLAlchemy, PyMySQL, mysql-connector-python |
| Web app | Streamlit |
| Environment | `carVenv` (virtual environment created from the terminal) |
| IDE | VS Code, Jupyter Notebook |

---

## 📁 Project Structure

```
CarPricePredicationProject/
│
├── app/
│   └── car_price_app.py          # Streamlit web app
│
├── data/
│   ├── cars_dataset.csv          # Raw dataset
│   └── cleaned_cars_dataset.csv  # Cleaned dataset used for modelling
│
├── saved_models/                 # Trained models (pickle files)
│   ├── LinearRegression.pkl
│   ├── SVR.pkl
│   ├── DecisionTreeRegressor.pkl
│   ├── RandomForestRegressor.pkl # Model used by the app
│   ├── Ridge.pkl
│   └── Lasso.pkl
│
├── saved_scaling/
│   └── scaler.pkl                # Fitted StandardScaler
│
├── screenshots/                  # Images used in this README
│   ├── 01_form_top.png
│   ├── 02_vehicle_details.png
│   └── 03_prediction_result.png
│
├── Car_price_SQL_Work.sql        # SQL work
├── main.ipynb                    # Full ML pipeline notebook
├── requirements.txt              # Python dependencies
└── README.md
```

> The `carVenv/` folder (virtual environment) is created locally by you and is **not** part of the repository.

---

## 📊 Dataset

- **Source:** [mohitmahiyt/cardataset](https://github.com/mohitmahiyt/cardataset) (`cars_dataset.csv`)
- **Size after cleaning:** 15,244 rows and 13 columns
- **Coverage:** 32 brands and 121 car names (from Maruti Alto to Bentley and Rolls-Royce)
- **Target column:** `selling_price` (in ₹)

| Column | Type | Description |
|---|---|---|
| `car_name`, `brand`, `model` | Categorical | Car identification (dropped before training) |
| `vehicle_age` | Numeric | Age of the car in years |
| `km_driven` | Numeric | Total kilometres driven |
| `seller_type` | Categorical | Dealer / Individual / Trustmark Dealer |
| `fuel_type` | Categorical | CNG / Diesel / Electric / LPG / Petrol |
| `transmission_type` | Categorical | Automatic / Manual |
| `mileage` | Numeric | Fuel efficiency (km/l) |
| `engine` | Numeric | Engine capacity (CC) |
| `max_power` | Numeric | Maximum power (bhp) |
| `seats` | Numeric | Number of seats |
| `selling_price` | Numeric | **Target** – price in ₹ |

---

## 🔄 Project Workflow

**1. Data collection**
The raw CSV is downloaded from GitHub into the `data/` folder using `urllib`.

**2. Data cleaning**
- Read the file with the correct header row (`header=11`)
- Removed 390 completely empty rows
- Dropped the empty extra column and the unnamed index column
- Removed 167 duplicate rows
- Converted `vehicle_age`, `km_driven`, `engine` and `seats` to integers
- Reset the index and saved the result as `cleaned_cars_dataset.csv`

**3. MySQL integration**
The cleaned DataFrame is sent to a MySQL database (`car_price_db`, table `cars`) using SQLAlchemy and PyMySQL, then read back with `pd.read_sql` to verify (15,244 rows and 13 columns).

**4. Exploratory Data Analysis**
- Univariate: KDE plots for numeric columns, count plots for categorical columns
- Outlier detection with box plots
- Bivariate: scatter plots of every numeric feature against `selling_price`
- Multivariate: correlation heatmap

**5. Feature engineering**
- Dropped `car_name`, `brand` and `model`
- One-hot encoded `seller_type`, `fuel_type` and `transmission_type`
- Train/test split of 80/20 (`random_state=0`)
- Scaled features with `StandardScaler`
- Checked feature importance with `ExtraTreesRegressor`

**6. Model building and evaluation**
Trained and compared Linear Regression, SVR, Decision Tree, Random Forest, Ridge and Lasso using MAE, MSE, RMSE, Explained Variance and R².

**7. Hyperparameter tuning**
Tuned every model with `GridSearchCV` (5-fold cross validation, scoring = R²).

**8. Saving artifacts**
Saved all models to `saved_models/` and the scaler to `saved_scaling/scaler.pkl`.

**9. Deployment**
Built a Streamlit app (`app/car_price_app.py`) that loads the Random Forest model and scaler and predicts the price from user input.

---

## 🏆 Model Results

Results on the test set (20% of the data) after hyperparameter tuning:

| Model | MAE (₹) | RMSE (₹) | R² Score |
|---|---|---|---|
| Linear Regression | 272,203 | 504,243 | 0.6935 |
| SVR | 324,486 | 816,394 | 0.1966 |
| Decision Tree | 135,914 | 526,399 | 0.6660 |
| **Random Forest** | **103,915** | **226,843** | **0.9380** |
| Ridge | 272,064 | 504,320 | 0.6934 |
| Lasso | 272,203 | 504,243 | 0.6935 |

**Best model: Random Forest Regressor**

- Best parameters: `n_estimators=50`, `max_depth=20`, `min_samples_split=2`
- The saved model scores about **0.97 R² on training data** and **0.93 R² on test data**, so it generalises well.

---

## ⚙ Installation and Setup (Terminal Commands)

All commands below are for **Windows (Command Prompt or VS Code terminal)**.

### Step 1: Clone the repository

```bash
git clone https://github.com/AnuTech-01/Car-Price-Predication-Project.git
cd Car-Price-Predication-Project
```

### Step 2: Create the virtual environment (`carVenv`)

**Option A: using Conda (recommended, Python 3.10)**

```bash
conda create -p carVenv python=3.10 -y
```

**Option B: using Python venv (needs Python 3.10 installed)**

```bash
python -m venv carVenv
```

### Step 3: Activate the environment

**If you used Conda:**

```bash
conda activate .\carVenv
```

**If you used venv:**

```bash
carVenv\Scripts\activate
```

After activation, the terminal line starts with `(carVenv)`.

### Step 4: Install the dependencies

```bash
pip install -r requirements.txt
```

If you want to install the libraries manually instead:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn six streamlit sqlalchemy pymysql mysql-connector-python jupyter ipykernel
```

### Step 5: Register the environment as a Jupyter kernel (optional)

Needed only if you want to open `main.ipynb` and choose `carVenv` as the kernel in VS Code or Jupyter:

```bash
python -m ipykernel install --user --name carVenv --display-name "carVenv (Python 3.10)"
```

### Step 6: Run the notebook (optional)

Open `main.ipynb`, select the `carVenv` kernel and click **Run All**. This will:

1. Download and clean the dataset
2. Perform EDA
3. Train and tune all models
4. Save the models to `saved_models/` and the scaler to `saved_scaling/`

> **MySQL cells:** the notebook contains a section that uploads data to a local MySQL server. To run it you need MySQL installed and running on `127.0.0.1:3306`, and you must enter **your own** MySQL username and password in those cells. If you do not use MySQL, skip that section.
>
> The trained `.pkl` files are already included, so you can skip this step and run the app directly.

---

## ▶ Run the Streamlit App

From the project root folder (with `carVenv` activated):

```bash
streamlit run app\car_price_app.py
```

If you do not want to activate the environment, run it directly with the environment's Python:

```bash
carVenv\python.exe -m streamlit run app\car_price_app.py
```

(For a venv created with `python -m venv`, use `carVenv\Scripts\python.exe` instead.)

The app opens automatically in your browser at:

```
http://localhost:8501
```

**To stop the app:** press `Ctrl + C` in the terminal.

**To run on a different port** (if 8501 is busy):

```bash
streamlit run app\car_price_app.py --server.port 8502
```

**First run notes**

- Streamlit may ask for an email address. This is optional, so just press `Enter`.
- Windows Firewall may show a popup. For local use you can click **Cancel**, because `localhost` works without it.

---

## 🖥 How to Use the App

1. Enter the car details:
   - Vehicle Age (years)
   - KM Driven
   - Mileage (km/l)
   - Engine (CC)
   - Max Power (bhp)
   - Seats
2. Select the **Seller Type**, **Fuel Type** and **Transmission Type**.
3. Click **Predict Price**.
4. The estimated price appears in a green box, in Lakhs or Crores.
5. Click **Model Input Features** to see the exact values sent to the model.

**Example**

| Field | Value |
|---|---|
| Vehicle Age | 7 |
| KM Driven | 94,945 |
| Mileage | 24.79 |
| Engine | 1197 CC |
| Max Power | 47 |
| Seats | 5 |
| Seller Type | Individual |
| Fuel Type | Petrol |
| Transmission | Manual |

**Predicted price: ₹ 2.32 Lakhs**

> The **Car Name** field is only for reference. It is not used by the model, because `car_name`, `brand` and `model` were dropped during feature engineering.

---

## 🔢 Model Input Features

The model expects **16 features in exactly this order** (the same order used during training):

| # | Feature | Type |
|---|---|---|
| 1 | `vehicle_age` | Numeric |
| 2 | `km_driven` | Numeric |
| 3 | `mileage` | Numeric |
| 4 | `engine` | Numeric |
| 5 | `max_power` | Numeric |
| 6 | `seats` | Numeric |
| 7 | `seller_type_Dealer` | One-hot (0/1) |
| 8 | `seller_type_Individual` | One-hot (0/1) |
| 9 | `seller_type_Trustmark Dealer` | One-hot (0/1) |
| 10 | `fuel_type_CNG` | One-hot (0/1) |
| 11 | `fuel_type_Diesel` | One-hot (0/1) |
| 12 | `fuel_type_Electric` | One-hot (0/1) |
| 13 | `fuel_type_LPG` | One-hot (0/1) |
| 14 | `fuel_type_Petrol` | One-hot (0/1) |
| 15 | `transmission_type_Automatic` | One-hot (0/1) |
| 16 | `transmission_type_Manual` | One-hot (0/1) |

The input is scaled with the saved `StandardScaler` and then passed to the Random Forest model.

---

## 🧯 Troubleshooting

| Problem | Solution |
|---|---|
| `streamlit is not recognized` | Activate `carVenv` first, or run `pip install streamlit` inside it |
| `carVenv\Scripts\activate` not recognized | Your environment was created with Conda. Use `conda activate .\carVenv` instead |
| `ModuleNotFoundError: No module named ...` | Activate `carVenv` and run `pip install -r requirements.txt` |
| `FileNotFoundError` for the `.pkl` files | Run the app from the project root and make sure `saved_models/` and `saved_scaling/` exist |
| Warning about scikit-learn version mismatch when loading `.pkl` | Install the same scikit-learn version used for training, or re-run `main.ipynb` to regenerate the `.pkl` files |
| `NoSessionContext` or `missing ScriptRunContext` | Do not run Streamlit code inside a notebook. Use `streamlit run` from the terminal |
| Port 8501 already in use | Use `--server.port 8502` |
| Wrong or very odd predictions | Check that the input order matches the table in [Model Input Features](#-model-input-features) |

---

## 🚀 Future Improvements

- Use car brand and model as input features
- Add more data and more recent listings
- Try Gradient Boosting, XGBoost or LightGBM
- Show a price range instead of a single value
- Deploy the app on Streamlit Community Cloud
- Add input validation and better error messages

---

## 👤 Author



**Anu Jangid**
GitHub: (https://github.com/AnuTech-01)
Linkedin : (https://www.linkedin.com/in/anu-jangid-726564328/)

If you found this project useful, please give it a ⭐ on GitHub.
