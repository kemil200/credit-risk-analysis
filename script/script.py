import pandas as pd
import numpy as np

np.random.seed(42)
n_dossiers = 1000

data = {
    'ID_Membre': range(1001, 1001 + n_dossiers),
    'Age': np.random.randint(21, 65, n_dossiers),
    'Secteur_Activite': np.random.choice(['Agriculture', 'Commerce', 'Transport', 'Services'], n_dossiers),
    'Anciennete_Coop_Mois': np.random.randint(6, 120, n_dossiers),
    'Revenu_Mensuel_Moyen': np.random.randint(150000, 1500000, n_dossiers),
    'Montant_Pret': np.random.randint(500000, 5000000, n_dossiers),
    'Duree_Pret_Mois': np.random.choice([6, 12, 18, 24], n_dossiers),
    'Epargne_Disponible': np.random.randint(50000, 1000000, n_dossiers),
}

df = pd.DataFrame(data)
df['Ratio_Endettement'] = (df['Montant_Pret'] / df['Duree_Pret_Mois']) / df['Revenu_Mensuel_Moyen']
df['Ratio_Solvabilite'] = df['Epargne_Disponible'] / df['Revenu_Mensuel_Moyen']
df['Indice_Fidelite'] = df['Anciennete_Coop_Mois'] / df['Duree_Pret_Mois']

# Simulation de la cible
score_risque = (df['Ratio_Endettement'] * 5) - (df['Anciennete_Coop_Mois'] / 100) + np.random.normal(0, 0.5, n_dossiers)
df['Statut_Defaut'] = (score_risque > score_risque.quantile(0.85)).astype(int)

df.to_csv('dossiers_prets_coop.csv', index=False)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('dossiers_prets_coop.csv')
df = pd.get_dummies(df, columns=['Secteur_Activite'], prefix='Secteur')

features = ['Age', 'Revenu_Mensuel_Moyen', 'Montant_Pret', 'Duree_Pret_Mois', 
            'Ratio_Endettement', 'Ratio_Solvabilite', 'Indice_Fidelite',
            'Secteur_Agriculture', 'Secteur_Commerce', 'Secteur_Services', 'Secteur_Transport']

X = df[features]
y = df['Statut_Defaut']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)



from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def generer_rapport(df, model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    
    print(f"--- RAPPORT DE PERFORMANCE ---")
    print(f"Précision du modèle : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Score AUC : {auc:.2f}")
    
    # Visualisation de la distribution du risque
    sns.kdeplot(df[df['Statut_Defaut']==0]['Ratio_Endettement'], label='Sain', fill=True)
    sns.kdeplot(df[df['Statut_Defaut']==1]['Ratio_Endettement'], label='Défaut', fill=True)
    plt.title('Inférence : Ratio Endettement vs Risque')
    plt.legend()
    plt.show()

generer_rapport(df, model, X_test_scaled, y_test)



