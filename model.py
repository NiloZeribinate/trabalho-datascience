import pandas as pd
import numpy as np

from sklearn import tree, set_config  # Arvore de decisão e plot tree
from sklearn.metrics import accuracy_score   # Acurácia
from sklearn.compose import ColumnTransformer # Tratar as colunas nominais de forma unificada
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, KBinsDiscretizer  # Transformar coluna ordinária
from sklearn.model_selection import train_test_split  # Separar a parte de teste
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Matriz de confusão
import matplotlib.pyplot as plt  # Plot na tabela
from sklearn.impute import KNNImputer

set_config(transform_output="pandas")

df=pd.read_csv("class_german_credit.csv")

# Sex
df['Sex'] = (df['Sex'] == 'male').astype(int) # female -> 0; male -> 1;

# Colunas nominais
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

print(df)

# Risk
df['Risk'] = (df['Risk'] == 'good').astype(int) # bad -> 0; good -> 1;

encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value= -1,categories=[['little', 'moderate', 'rich']])
imputer = KNNImputer(n_neighbors=5) #inicializando kNN e encoder

X = df.drop('Risk', axis=1)
y = df['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify = y)

max=-1

for i in range(2,20):
    age_discretizer = KBinsDiscretizer(
        n_bins=i,
        encode='ordinal',
        strategy='uniform'
    )
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    # usar copias

    X_train_copy['Age'] = age_discretizer.fit_transform(X_train[['Age']])
    X_test_copy['Age'] = age_discretizer.transform(X_test[['Age']])
    for j in range(2,20):
        credit_discretizer = KBinsDiscretizer(n_bins=j,encode='ordinal',strategy='uniform')
        X_train_copy['Credit amount'] = credit_discretizer.fit_transform(X_train[['Credit amount']])
        X_test_copy['Credit amount'] = credit_discretizer.transform(X_test[['Credit amount']])
        clf = tree.DecisionTreeClassifier(class_weight='balanced',random_state=42)

        # Treinamento
        clf.fit(X_train_copy, y_train)

        # Teste
        y_pred = clf.predict(X_test_copy)

        acuracia = accuracy_score(y_test, y_pred)
        if acuracia>max:
            max=acuracia
            print(f'A acurácia do modelo foi de {acuracia*100:.2f}% com {i} bins de idade e {j} bins de credit amount')

cm = confusion_matrix(y_test, y_pred)
tree.plot_tree(clf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão')


importancia = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Importância das colunas:\n", importancia)

plt.show()