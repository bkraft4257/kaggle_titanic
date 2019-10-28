Executive Summary
=================

`Titanic: Machine Learning from Disaster <https://www.kaggle.com/c/titanic/overview>`_

**Goal:**
    Create a machine learning model that predicts which passengers
    survived the Titanic shipwreck.
    
    Accuracy goal of 82% on the Holdout Kaggle Competition data set.





**Best Model:**
    Logistic Regression with Age  81.8182%

Current Status
--------------
Oct 17, 2019 at 2:30:03 PM

**Situation**
    Completed first model with simple Logistic Regression.  The model
    achieved an accuracy of 80%.

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
.. table::

    ============================= ======================== ===================================
    Model                         Accuracy                 Link
    ============================= ======================== ===================================
    logreg                        0.803370                 :ref:`model-logreg`
    logreg_with_age               0.804195                 :ref:`model-logreg_with_age`
    logreg_with_age_scaled        0.811188                 :ref:`model-scaled_logreg_with_age`
    logreg_with_ticket_scaled     0.818182
    ============================= ======================== ===================================
