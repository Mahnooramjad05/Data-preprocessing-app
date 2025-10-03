🔬 Interactive Machine Learning Pipeline with ipywidgets

This project is an interactive Jupyter Notebook interface for building end-to-end Machine Learning pipelines using ipywidgets.
It allows you to upload data, preprocess features, visualize graphs, choose models (Regression/Classification), train, evaluate, cross-validate, tune hyperparameters, and make predictions — all without writing additional code.

🚀 Features
📥 Data Collection

Upload CSV datasets interactively.
View dataset information: number of rows, columns, and first 5 records.
Auto-detect target column and feature columns.
Handle missing values (Mean, Median, Mode, Forward Fill, Backward Fill).

📊 Visualization
Multiple chart options:
Line, Bar, Column, Pie, Scatter, Histogram, Box, Area, Bubble, Heatmap.

⚙️ Data Preprocessing
Drop columns, handle missing values, and encode categorical features.
Apply scaling methods: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer.
Visualize scaling effects using histograms.

🧠 Model Selection
Choose between:
Regression Models: Linear Regression, Ridge, Lasso, Elastic Net.
Classification Models: Logistic Regression, KNN, SVM, Decision Tree, Random Forest.

📈 Training & Evaluation
Train models with train-test split.

Evaluate:
Regression → Mean Squared Error (MSE), R² Score.
Classification → Accuracy, Classification Report.
Graphical evaluation with training vs. testing results.

🔄 Cross-Validation
Supports K-Fold, Stratified K-Fold, and Leave-One-Out cross-validation.
Displays mean and standard deviation of scores.

🎯 Hyperparameter Tuning
Automated Grid Search CV for all models.
Finds best parameters and cross-validation score.

🤖 Predictions & Deployment
Auto-generated input widgets for new feature values.
Make predictions interactively with trained models.
Decodes categorical predictions if label encoding is applied.

📦 Installation
Clone this repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Install dependencies:
pip install -r requirements.txt

📂 File Structure
📁 your-repo-name
 ┣ 📜 notebook.ipynb        # Main Jupyter Notebook with ipywidgets pipeline
 ┣ 📜 requirements.txt      # Dependencies
 ┗ 📜 README.md             # Documentation

⚙️ Requirements
Python 3.8+
Jupyter Notebook / JupyterLab
Libraries:
pandas, numpy
matplotlib, seaborn
scikit-learn
ipywidgets
scipy

Install everything with:
pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets scipy


Enable widgets in Jupyter:
jupyter nbextension enable --py widgetsnbextension

🎯 Example Workflow
Upload your dataset (CSV).
Explore data info & missing values.
Visualize relationships with graphs.
Apply preprocessing (scaling, encoding, imputation).
Select target and features.
Choose model type → Regression or Classification.

Train the model.

Evaluate with metrics & visualizations.
Apply cross-validation.
Tune hyperparameters with GridSearchCV.
Deploy prediction form → input new values and predict interactively.

🛠️ Future Enhancements
Add support for Deep Learning models (LSTM, GRU, CNN).
Export trained models and predictions.
Add AutoML-style pipeline automation.
Interactive report export (PDF/HTML).

👨‍💻 Author

Developed by Mahnoor Amjad✨
