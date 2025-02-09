# FASHION-CLASSIFIER

Implementació de K-means i KNN per tal de classificar peces de roba segons la seva forma i color.

## Contingut del repositori

Aquest repositori inclou:

- **images**: Carpeta que conté les imatges utilitzades. Dins d'aquesta carpeta:
  - **test**: Conjunt d’imatges que farem servir com a conjunt de test.
  - **train**: Conjunt d’imatges que utilitzarem com a conjunt d’entrenament per a la classificació de formes.
  - **gt.json**: Arxiu amb la informació del Ground-Truth de les imatges.
  - **gt_reduced**: Arxiu amb informació complementària sobre una part de les imatges que conformen el training set.

- **utils.py**: Conté una sèrie de funcions necessàries per a convertir les imatges en color en altres espais.

- **Kmeans.py**: Arxiu amb les funcions necessàries per a implementar el K-means per extreure els colors predominants.

- **KNN.py**: Arxiu amb les funcions necessàries per implementar KNN i per etiquetar el nom de la peça de roba.

- **my_labeling.py**: Arxiu on es combinen els dos mètodes d’etiquetatge i les millores per obtenir l’etiquetatge final de les imatges. Les diferents proves estan comentades i indicades.

- **utils_data.py**: Conté una sèrie de funcions necessàries per a la visualització de resultats.

- **Informe**: Informe amb les proves detallades i els resultats obtinguts.

## INSTAL·LACIÓ 

1. **Clona el repositori:**
   ```bash
   git clone https://github.com/MMC26/Fashion-Classifier.git
   cd Fashion-Classifier
   ```
2. **Llibreries:**
    Aquest projecte utilitza diverses llibreries de Python que cal instal·lar abans d’executar-lo. Pots fer-ho creant un entorn virtual i instal·lant-les amb pip:

    ```bash
    pip install numpy
    pip install matplotlib
    pip install scipy
    ```

## RESULTATS
- **Kmeans**
    - Millor mètode d'inicialització: Kmeans++
    - Millor heurística per bestK: Fisher Discriminant
    - Major accuracy pels colors: Distància intra-class
    - Millor K: K=6 amb 80% d'accuracy
- **KNN** 
    - Millors distàcnies: Manhattan i Minkowski major accuracy, però Euclidiana més ràpida.
    - Millor K: K=4 amb 90% d'accuracy

## AUTORS
- David Ruiz Cáceres
- Berta Martí Boncompte
- Blanca Pinyol Chacon 
- Maria Muñoz Cabestany 