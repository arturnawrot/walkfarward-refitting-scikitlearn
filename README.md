### Description

Use a cumulitive prediction from multiple ai models. Retrains your models with most recent n candles (useful for walkfarward backtesting). 

### Usage

```python
walkforward = Walkforward()

walkforward.add_model(
    KNeighborsClassifier(n_neighbors=4, random_state=0),
    ['Close', 'X_RSI', 'X_SMA50', 'rsi_sma']
)

walkforward.add_model(
    KNeighborsClassifier(n_neighbors=21, random_state=0),
    ['Close', 'X_RSI', 'X_SMA50']
)

walkforward.add_model(
    RandomForestClassifier(...),
    [...]
)

walkforward.fit_models(data_to_train)
y_pred = walkforward.get_walkfarward_prediction(data_to_test, 500, 100)
```
