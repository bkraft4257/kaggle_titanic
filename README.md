kaggle_titanic
==============

How do I become a more agile data scientist?

Fundamentally, that is the question that I am trying to answer. The Kaggle
Titanic Tutorial competition is just the first project that I have chosen
to figure out how to answer the question above.

If I am going to an agile data scientist I must work in small iterative steps.
While I have hundreds of ideas on how to improve, for me taking the first step
was more important than taking the right step.  After all, if I am really agile
I can adjust my direction as I learn more about the process.

I do have some guiding principles that I believe can help me with the start of
my agile journey.

1. Extracting actionable insights from data to create business value (not
creating models) is the job of a data scientist.
2. The collective intelligence of the team is smarter than a single person.
3. Sharing knowledge among team members leads to actionable insights
4. Creating shareable knowledge can be done in small iterative batches.
5. Use existing tools and algorithms effectively to quickly create
business value.

In the spirit of Principle #5, I created this repository from the Data Science
Cookie Cutter

https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e

The project organization was created for me.  I will slowly adapt this cookie
cutter for my needs but it was a great place to start.  It provided my project
some structure and set up the Sphinx environment for my project.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
