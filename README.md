# Machine Learning project-1: Team SES

The  Higgs  boson  is  an  elementary  particle  thatexplains  why  other  particles  have  mass.  It  is  important  to  beable  to  classify  whether  the  particle  decay  signature  belongsto the Higgs boson, because it fills other particles with a massconfirming  the  physical  theory.  In  this  report  we  introducesix  methods  for  this  classification  task.  After  the  trainingand  validation  of  our  classifiers,  we  achieved  the  a82.3%ofaccuracy  and72.9%of  F1  score  with  the  Ridge  regressionon  the  official  testing  platform.  We  show  that  applying  someadditional pre-processing of the dataset and polynomial featureexpansion  improve  our  predictions.

# Scripts:

run.py  - file for counting base model results.
Example
```
python run.py -path ../data/train.csv
```
BaselineModels.ipynb - notebook for analysis for base models.

DataPreprocessedModels.ipynb  - notebook with dataprocessing.
