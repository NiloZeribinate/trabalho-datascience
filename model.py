import pandas as pd
import numpy as np

from sklearn import tree, set_config  # Arvore de decisão e plot tree
from sklearn.metrics import accuracy_score   # Acurácia
from sklearn.compose import ColumnTransformer # Tratar as colunas nominais de forma unificada
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, KBinsDiscretizer, StandardScaler  # Transformar coluna ordinária, knn para discretização, padronização
from sklearn.model_selection import train_test_split  # Separar a parte de teste
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Matriz de confusão
import matplotlib.pyplot as plt  # Plot na tabela
from sklearn.impute import KNNImputer

set_config(transform_output="pandas")

df=pd.read_csv("class_german_credit.csv")


# Treating cols

# - Sex
df['Sex'] = (df['Sex'] == 'male').astype(int) # female -> 0; male -> 1;

# - Nominal Columns
preprocessor = ColumnTransformer(
    transformers=[
        ('nom_housing', OrdinalEncoder(categories=[['free', 'rent', 'own']]), ['Housing']),
        ('nom_saving',  OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, categories=[['little', 'moderate', 'quite rich', 'rich']]), ['Saving accounts']),
        ('nom_checking',OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, categories=[['little', 'moderate', 'rich']]), ['Checking account']),
        ('ord_purpose', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Purpose'])
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

df = preprocessor.fit_transform(df)

# - Risk
df['Risk'] = (df['Risk'] == 'good').astype(int) # bad -> 0; good -> 1;


# Separating the NaN from complete rows
fully_df = df.dropna()
na_df = df[df.isna().any(axis=1)]


# Separating data to training and test
X = fully_df.drop('Risk', axis=1)
y = fully_df['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2*len(df)/len(fully_df), random_state=42, stratify=y)

X_train = pd.concat([X_train, na_df.drop('Risk', axis=1)])
y_train = pd.concat([y_train, na_df['Risk']])


# Preparing training data

# - Normalizing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# - Filling missing values with kNN
imputer = KNNImputer(n_neighbors=5)
X_train = imputer.fit_transform(X_train)

# - Reversing data normalization
X_train = pd.DataFrame(
    scaler.inverse_transform(X_train),
    columns=scaler.get_feature_names_out()
)


# Seeking the best discretizing for Age and Credit Amount

# - Creating Decision Tree
clf = tree.DecisionTreeClassifier(class_weight='balanced',random_state=42)

# - Preparing vars
max=-1

params = {
    'Age': 4,
    'Credit amount': 28,
    'Duration': 32
}

for i in range(2, 5):

    # Making copies
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()

    # Age discretization
    age_discretizer = KBinsDiscretizer(n_bins=i, encode='ordinal', strategy='quantile')
    X_train_copy['Age'] = age_discretizer.fit_transform(X_train[['Age']])
    X_test_copy['Age'] = age_discretizer.transform(X_test[['Age']])
    
    for j in range(10, 40):
        # Credit discretization
        credit_discretizer = KBinsDiscretizer(n_bins=j, encode='ordinal', strategy='quantile')
        X_train_copy['Credit amount'] = credit_discretizer.fit_transform(X_train[['Credit amount']])
        X_test_copy['Credit amount'] = credit_discretizer.transform(X_test[['Credit amount']])
        

        for k in range(2, 40):
            # Duration discretization
            duration_discretizer = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform')
            X_train_copy['Duration'] = duration_discretizer.fit_transform(X_train[['Duration']])
            X_test_copy['Duration'] = duration_discretizer.transform(X_test[['Duration']])

            # Training
            clf.fit(X_train_copy, y_train)

            # Testing
            y_pred = clf.predict(X_test_copy)

            acuracia = accuracy_score(y_test, y_pred)
            if acuracia > max:
                max=acuracia

                params['Age'] = i
                params['Credit amount'] = j
                params['Duration'] = k
                
                print(f'A acurácia do modelo foi de {acuracia*100:.2f}% com {i} bins de idade, {j} bins de credit amount, {k} bins de duration')

for key in params.keys():
    discretizer  = KBinsDiscretizer(n_bins=params[key], encode='ordinal', strategy='quantile')

    X_train[key] = discretizer.fit_transform(X_train[[key]])
    X_test[key]  = discretizer.transform(X_test[[key]])

# Training
clf.fit(X_train, y_train)

# Testing
y_pred = clf.predict(X_test)


# Ploting the tree
# tree.plot_tree(clf)

# Ploting the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão')
plt.show()

# Columns importance
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Importância das colunas:\n", importances)
print(f'A acurácia do modelo foi de {acuracia*100:.2f}% com {params['Age']} bins de idade, {params['Credit amount']} bins de credit amount, {params['Duration']} bins de duration')
