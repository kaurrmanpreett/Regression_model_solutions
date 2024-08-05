from .import_module import *

def train_linear_regression(x_train, y_train):
    lrmodel = LinearRegression().fit(x_train, y_train)
    return lrmodel

def train_decision_tree(x_train, y_train):
    dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567)
    dtmodel = dt.fit(x_train, y_train)
    return dtmodel

def train_random_forest(x_train, y_train):
    rf = RandomForestRegressor(n_estimators=200, criterion='absolute_error')
    rfmodel = rf.fit(x_train, y_train)
    return rfmodel

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model
