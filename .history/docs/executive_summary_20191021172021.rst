Executive Summary
=================

`Titanic: Machine Learning from Disaster <https://www.kaggle.com/c/titanic/overview>`_

**Goal:**
    Create a machine learning model that predicts which passengers
    survived the Titanic shipwreck.

    A model will be determined successful if it can achieve an 82% accuracy
    rating on the Holdout Kaggle Competition data set.

**Best Model:**
    Logistic Regression with Age  81.8182%

Current Status
--------------
Oct 17, 2019 at 2:30:03 PM

**Situation**
    I am able to achieve a 0.82 accuracy score on the test data set. However, 
    when I calculate the accuracy on the Kaggle Holdout data my score drops to 
    0.79.

**Background**
    First model did not use age, name, and pclass in the models.

**Assessment**
    Age is known for 80% of the participants. Include age in the model. When
    age is NaN fill with the mean age of all the passengers.

    Logistic Regression with Age did not change the accuracy of the model.

**Recommendation**
    Try Normalizing the features before fitting.

Models
------

.. csv-table::
   :file: ./_tables/model_accuracy.csv
   :header-rows: 1
   :widths: 8, 20, 20, 20,20, 10, 30


This is a table.
