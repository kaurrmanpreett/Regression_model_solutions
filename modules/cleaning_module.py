from .import_module import *

    
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split_data(df):
    x = df.drop('price', axis=1) 
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test
