def evaluate_model(train_target, train_preds, validation_target, validation_preds):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    print('The Mean Absolute Error for training set is ', mean_absolute_error(train_target, train_preds))
    print('The Mean Absolute Error for validation set is ', mean_absolute_error(validation_target, validation_preds))
    
    print('The Root Mean Squared Error for training set is ', mean_squared_error(train_target, train_preds, squared=False))
    print('The Root Mean Squared Error for validation set is ', mean_squared_error(validation_target, validation_preds, squared=False))