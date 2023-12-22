# modele_de_scoring

*Projet développé dans le cadre de la formation Data Scientist OC (RNCP niveau 7)*

La société financière "Prêt à dépenser" propose des crédits à la consommation pour des personnes ayant peu ou pas d'historique de prêt.
L’entreprise souhaite mettre en œuvre **un outil de “scoring crédit”** pour (1) calculer la probabilité qu’un client rembourse son crédit et (2) classifier la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.)

source des données : https://www.kaggle.com/c/home-credit-default-risk/data

travaux d'analyse des données : [01_EDA.ipynb](01_EDA.ipynb)

## Objectif:
- Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique ([02_Modelisation.ipynb](02_Modelisation.ipynb) & [best_xgb_1_2.joblib](best_xgb_1_2.joblib))
- Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, dans un soucis de transparence, de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle ([02_Modelisation.ipynb](02_Modelisation.ipynb) & [explainer_xgb_1_2.joblib](explainer_xgb_1_2.joblib))
- Mettre en production le modèle de scoring de prédiction à l’aide d’une API ([cf. repository de l'API](https://github.com/estellec18/app_credit_scoring)) et réaliser une interface de test de cette API ([frontend.py](frontend.py))
- Mettre en oeuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à l’analyse en production du data drift


Fonctionnement de l'interface :
- l'utilisateur choisit (dans une liste déroulante) le numéro du client dont il souhaite connaitre les résultats
- l'utilisateur clique sur le bouton "Prédiction" pour générer :
    - des informations générales sur le client en question (sexe, revenu, occupation...)
    - la probabilité de défault du client ainsi que sa classe (demande de credit acceptée ou refusée)
    - la visualisation du score du client sur une jauge
    - des informations concernant les principales features responsables du score et le positionnement du client par rapport au reste de la population

[lien vers le repository du dashboard](https://github.com/estellec18/dashboard_credit_scoring)

Nous avons également testé l’utilisation de la librairie evidently pour détecter du Data Drift en production (avec le dataset “application_train” représentant les datas pour la modélisation et le dataset “application_test” représentant les datas de nouveaux clients une fois le modèle en production).
- rapport de data drift généré par evidently [data_drift_report.html](data_drift_report.html)
