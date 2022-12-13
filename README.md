# Compétition Kaggle 
## Natural Language Processing with Disaster Tweets
<img src= "https://storage.googleapis.com/kaggle-media/competitions/nlp1-cover.jpg" > 

### Introduction
Kaggle est un site de data science organisant des compétitions intéressantes à des buts d’apprentissage ou de résolution des problèmes réels de la vie.
En tant qu'étudiant Ingénieur Machine Learning, 
Il nous est demandé de participer à une compétition Kaggle de notre choix.
Nous avons donc choisi de participer à ma compétition « Natural Language Processing with Disaster Tweets » qui est une compétition en traitement naturel de Langage.
Ce projet social rend service énormément à la communauté car permettra aux secouristes d'être alertés en cas de catastrophe à un endroit et donc vite intervenir, ce qui a suscité notre grand intérêt porté à cette compétition.


### Plan de notre travail
•	Nettoyage et analyse exploratoire
•	Feature extractions
•	Test de différents modèles avec les différentes méthodes d’extraction de features utilisé
•	Utilisation d’une métrique commune pour mesurer les résultats et comparaison
•	Soumission du résultat de chaque model sur Kaggle

### Bibliothèques:
    * transformers
    * sckit-learn
    * tensorflow
    * pandarallel
    * pandas
    * numpy
    * seaborn
    * wordcloud
    * gensim
    * nltk
    * spacy

### A. Prétraitement du Test

Nous nettoyons le texte et nous le transformons :
Suppression:
      •	URL
      •	Balises HTML
      •	Références de personnages
      •	Caractères non imprimables
      •	Valeurs numériques
      Traitement
      •	Lemmatisons le texte 
      •	Conversion en minuscules. 
      •	Suppression des caractères répétés dans les mots allongés,
      •	Suppression des mots vides
      •	Conservation des hashtags car ils peuvent fournir des informations précieuses sur ce projet particulier.



#### Feature Engineering

Nous créons 10 colonnes qui sont :
   •	Nombre de phrases
   •	Nombre de mots
   •	Nombre de caractères
   •	Nombre de hashtags
   •	Nombre de mentions
   •	Nombre de mots tout en majuscules
   •	Longueur moyenne des mots
   •	Nombre de noms propres (PROPN)
   •	Nombre de noms non propres (NOM)
   •	Pourcentage de caractères qui sont de la ponctuation

### B. Features extractions 

Cette étape nous permet de convertir les documents en vecteur pour etre utilisé par les modèles pour la prédiction
Nous utilisons 2 méthodes principales :

   *	TfidfVectorizer
   *	Doc2vec 



### C.	Modélisation : Résultats


#### 1.	Logistic Régression :

##### a.	With TFIDFVectorizer 

   •	Nous recherchons les meilleurs hyper paramètres avec GridSearchCV
   •	Accuracy : 0.7809139784946236        
##### b. With Doc2vec
   •	Nous recherchons les meilleurs hyper paramètres avec GridSearchCV
   •	Accuracy : 0.6503311258278146


 #### 2. 	Transformers
##### a.	With TFIDFVectorizer 
   Accuracy: 0.569

##### b.	With Doc2vec
Accuracy: 0.558



  #### 3. Pre entrained model nnlm-en-dim50
  
            Accuracy: 0.7775
