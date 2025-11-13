Outil de Prévision des Sinistres en Assurance Automobile

Description du Projet

Outil de prévision des sinistres en assurance automobile utilisant des modèles statistiques et d'IA. Cette application web permet aux assureurs de prédire la fréquence et le coût des sinistres, calculer les primes appropriées, et gérer efficacement les portefeuilles clients dans le contexte burundais.

Fonctionnalités Principales

Gestion Multi-utilisateurs
Clients : 
Soumission de demandes, upload de documents, suivi des contrats
Agents : 
Traitement des demandes, calcul des prédictions, envoi de primes
Administrateurs : 
Supervision complète, gestion des utilisateurs, statistiques

Modèles de Prédiction
Random Forest: pour la fréquence et la sévérité des sinistres
Calcul automatique: des primes en Francs Burundais (BIF)
Probabilité de sinistre: basée sur la distribution de Poisson


Installation et Configuration

Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Navigateur web moderne

Installation

1. Cloner le dépôt
```bash
git clone https://github.com/votre-username/AutoAssurance.git
cd AutoAssurance
2.Créer un environnement virtuel
Windows
python -m venv venv
venv\Scripts\activate

 Linux/Mac
python3 -m venv venv
source venv/bin/activate
3.Installer les dépendances
pip install -r requirements.txt
4.Configurer l'application
Créer le dossier pour les uploads
mkdir uploads
5.Lancer l'application
python app.py
6.Accéder à l'application
Ouvrir http://localhost:5000 dans votre navigateur
