from modules.import_module import *
from modules.cleaning_module import *
from modules.Training_module import *
from modules.evaluation_module import *

# Load and Split Data
file_path = 'C:\\Users\\kaurr\\OneDrive\\Desktop\\BISI\\2208\\Regression model solutions\\data\\final.csv'
df = load_data(file_path)
x_train, x_test, y_train, y_test = split_data(df)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#Linear Regression Model
lrmodel = train_linear_regression(x_train, y_train)
lr_train_mae, lr_test_mae = evaluate_model(lrmodel, x_train, y_train, x_test, y_test)
print(f'Linear Regression - Train MAE: {lr_train_mae}, Test MAE: {lr_test_mae}')


#Decision Tree model
dtmodel = train_decision_tree(x_train, y_train)
dt_train_mae, dt_test_mae = evaluate_model(dtmodel, x_train, y_train, x_test, y_test)
plot_tree(dtmodel, 'tree.png')
print(f'Decision Tree - Train MAE: {dt_train_mae}, Test MAE: {dt_test_mae}')

#Random Forest model
rfmodel = train_random_forest(x_train, y_train)
rf_train_mae, rf_test_mae = evaluate_model(rfmodel, x_train, y_train, x_test, y_test)
print(f'Random Forest - Train MAE: {rf_train_mae}, Test MAE: {rf_test_mae}')

