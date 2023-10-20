# modele_de_scoring
*Projet 7 du parcours Data Science OC*

La société financière "Prêt à dépenser" propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.
L’entreprise souhaite mettre en œuvre **un outil de “scoring crédit”** pour (1) calculer la probabilité qu’un client rembourse son crédit et (2) classifier la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.)

source des données : https://www.kaggle.com/c/home-credit-default-risk/data

travaux d'analyse des données : [01_EDA.ipynb](01_EDA.ipynb)

## Missions:
- Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique ([02_Modelisation.ipynb](02_Modelisation.ipynb) & [best_xgb_1.joblib](best_xgb_1.joblib))
- Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, dans un soucis de transparence, de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle ([02_Modelisation.ipynb](02_Modelisation.ipynb) & [explainer_xgb_1.joblib](explainer_xgb_1.joblib))
- Mettre en production le modèle de scoring de prédiction à l’aide d’une API ([cf. repository de l'API](https://github.com/estellec18/app_credit_scoring)) et réaliser une interface de test de cette API ([frontend.py](frontend.py))
- Mettre en oeuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à l’analyse en production du data drift
