Doc strings
===========

API
---

.. automodule:: src.data.data
   :members: ExtractData, TransformData

.. automodule:: src.models.run
   :members: run_model


Internals
---------

.. automodule:: src.models.predict_model
   :members: concat_to_create_xy_test, calc_metrics, calc_logreg_model,
             calc_model_predictions, display_rst_table_metrics_log

.. automodule:: src.models.kaggle
   :members: submit_to_kaggle_titanic_competition, is_valid_kaggle_submission,
             upload_kaggle_titanic_submission_via_api

.. automodule:: src.models.run
   :noindex: 
   :members: argparse_command_line, read_X, read_y,
             write_kaggle_submission_output_file, predict, measure_accuracy
