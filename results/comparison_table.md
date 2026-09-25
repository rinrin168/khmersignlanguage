| approach    |   test_accuracy |   precision_macro |   recall_macro |   f1_macro |   trainable_params |   training_time_s |   epochs_run | device   |
|:------------|----------------:|------------------:|---------------:|-----------:|-------------------:|------------------:|-------------:|:---------|
| GRU         |            1    |          1        |       1        |   1        |             178827 |               5.8 |           48 | Tesla T4 |
| TRANSFORMER |            0.98 |          0.984848 |       0.977273 |   0.978749 |             307083 |               8.1 |           44 | Tesla T4 |
| LSTM        |            0.38 |          0.279221 |       0.409091 |   0.308678 |             236811 |               6.7 |           46 | Tesla T4 |