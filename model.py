import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree, set_config
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

# Configure sklearn to output pandas DataFrame
set_config(transform_output="pandas")

# Importing data
df = pd.read_csv("class_german_credit.csv")

df = df.drop(columns=['Purpose'])

# Splitting data
X = df.drop(columns='Risk')
y = df['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Preprocessing features
preprocessor = ColumnTransformer(
    transformers=[
        ('ord_sex', OrdinalEncoder(categories=[['female', 'male']]), ['Sex']),
        ('ord_housing', OrdinalEncoder(categories=[['free', 'rent', 'own']]), ['Housing']),
        ('ord_saving',  OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, categories=[['little', 'moderate', 'rich', 'quite rich']]), ['Saving accounts']),
        ('ord_checking',OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, categories=[['little', 'moderate', 'rich']]), ['Checking account']),
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

class ImputeInScaledSpace(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors = 5):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors = self.n_neighbors)

    def fit(self, X, y = None):
        X_scaled = self.scaler.fit_transform(X)
        self.imputer.fit(X_scaled)
        self.feature_names_out_ = self.scaler.get_feature_names_out()
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        X_imputed = self.imputer.transform(X_scaled)
        X_unscaled = pd.DataFrame(
            self.scaler.inverse_transform(X_imputed),
            columns = self.feature_names_out_
        )
        
        return X_unscaled

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

# Defining our pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('imputerInScale', ImputeInScaledSpace()),
    ('rounder', FunctionTransformer(np.round)),
    ('model', tree.DecisionTreeClassifier(class_weight='balanced', random_state=42)),
])


# Cross-Validation
param_grid = {
    'model__criterion': ['gini', 'entropy', 'log_loss'],
    'model__max_depth': [None, 2, 3, 4, 5, 6, 8, 10],
    'model__min_samples_split': [2, 5, 10, 15],
    'model__min_samples_leaf': [1, 2, 4, 6],
    'model__splitter': ['best', 'random'],
    'model__max_features': [None, 'sqrt', 'log2']
}

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

grid_search = GridSearchCV(
    estimator = pipeline,
    param_grid = param_grid,
    cv = cv,
    scoring = 'accuracy',
    n_jobs = -1,
    verbose = 1,
)

# Training model
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Testing model
y_pred = best_model.predict(X_test)


# Printing the best params
print(f"Best params: {grid_search.best_params_}")

# Evaluating the model acurracy
validation_accuracy = grid_search.best_score_
test_accuracy = accuracy_score(y_test, y_pred)

print(f'The accuracy of the model in validation was {validation_accuracy*100:.2f}% and {test_accuracy*100:.2f}% in test')

# Importance of features
feature_names = best_model[0].get_feature_names_out()
importances_values = best_model['model'].feature_importances_
importances = pd.Series(importances_values, index=feature_names).sort_values(ascending = False)

print("Features importances:\n", importances)

# Plotting the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = best_model['model'].classes_)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão')
plt.show()

# Plotting the tree decision tree

plt.figure(figsize=(20,10))

tree.plot_tree(
    best_model['model'], 
    filled = True, 
    feature_names = feature_names, 
    rounded = True,
    precision = 2,
    fontsize = 10
)

plt.show()
