import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree, set_config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

# Configure sklearn to output pandas DataFrame
set_config(transform_output="pandas")

# Importing data
df = pd.read_csv("class_german_credit.csv")

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
        ('nom_purpose', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Purpose']),
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

# Defining our pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', tree.DecisionTreeClassifier(class_weight='balanced', random_state=42))
])

# Training model
pipeline.fit(X_train, y_train)

# Evaluating the model acurracy
validation_accuracy = pipeline.score(X_test, y_test) # We don't have validation yet
test_accuracy = pipeline.score(X_test, y_test)

print(f'The accuracy of the model in validation was {validation_accuracy*100:.2f}% and {test_accuracy*100:.2f}% in test')

# Importance of features
feature_names = pipeline['preprocessing'].get_feature_names_out()
importances_values = pipeline['model'].feature_importances_
importances = pd.Series(importances_values, index=feature_names).sort_values(ascending = False)

print("Features importances:\n", importances)

# Plotting the Confusion Matrix
y_pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = pipeline['model'].classes_)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão')
plt.show()

# Plotting the tree decision tree

plt.figure(figsize=(20,10))

tree.plot_tree(
    pipeline['model'], 
    filled = True, 
    feature_names = feature_names, 
    rounded = True,
    precision = 2,
    fontsize = 10
)

plt.show()
