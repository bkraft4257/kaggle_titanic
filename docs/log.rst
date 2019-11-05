Log
===

Oct 16, 2019 at 8:31:52 PM
--------------------------
* Created project from Cookie Cutter.
* Downloaded data
* Ran Great Expectations Profiler.

Oct 17, 2019 at 12:52:51 PM
---------------------------

**Situation**
    First model needs to be built and tested.

**Background**
    The function for measuring the accuracy of the model was created.

**Assessment**
    The logistic regression used a very simple model. Better models
    with the same data may be able to improve the accuracy.  RandomForest
    decision trees may be a better approach as a classification problem.

**Recommendation**
    Try Logistic Regression with age as a feature.


Oct 17, 2019 at 2:30:03 PM
---------------------------

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


.. _log-current-status:

Oct 28, 2019 at 9:35:23 AM
--------------------------

    Documentation of this project needs updating and cleaning. This will be the
    first task today.

    Things that need attention:
    * Autodoc is not included in the Sphinx documentation.
    * Default Spinx theme should be replaced with Read The Docs theme.
    https://sphinx-rtd-theme.readthedocs.io/en/stable/ 

**Situation**
    I am able to achieve a 0.8384 accuracy score on the test data set. However,
    when I calculate the accuracy on the Kaggle Holdout data my score drops to
    0.79425.  WHY?

**Background**
    The Gender Only model produces a Holdout Accuracy score of 0.76555. The
    logreg_model_3 uses only 5 parameters ('title_Mr', 'title_Mrs',
    'family_size', 'is_child', 'pclass').  Reducing the number of features
    allowed the Kaggle Holdout Score to increase.

**Assessment**
    At this point, I believe we are reaching the point of diminishing returns
    with Logistic regression. From two perspectives,

    #. I don't believe the Kaggle Holdout accuracy score will increase
    that much. Others have reported higher accuracy scores but for
    they are reporting cross validation accuracy scores.

    #. I am limiting what I can learn by not using other models.

**Recommendation**
    I recommend exploring the use of Decision Trees for the Kaggle competition.



