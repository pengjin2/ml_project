from dimension_redux import DimensionRedux

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
import xgboost as xgb

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay


class FeaturePredictionMethods(DimensionRedux):
    def __init__(self):
        super().__init__()
        
    def check_data_integrity(self):
        # For Classifier model, y need to be discrete and label-like
        pass
    
    def _plot_classfier_with_title(custom_title=""):
        def decorator_plot_results(func):
            def wrapper(*args, **kwargs):
                # Execute the function
                X_test, y_test, predictions = func(*args, **kwargs)  # Function must return these values
                title_suffix = kwargs.get('title_suffix', custom_title)
                
                # Plotting confusion matrix for classifiers
                cm = confusion_matrix(y_test, predictions)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap='Blues')
                plt.title(f'{func.__name__}: Actual vs Predicted - {title_suffix}')
                plt.show()

                return X_test, y_test, predictions  # Make sure to return these values here as well
            return wrapper
        return decorator_plot_results

    
    # Define a visualizer to see the model fit 
    def _plot_regressor_with_title(custom_title=""):
        def decorator_plot_results(func):
            def wrapper(*args, **kwargs):
                # Execute the function
                X_test, y_test, predictions = func(*args, **kwargs)

                title_suffix = kwargs.get('title_suffix', custom_title)  # Get custom title from kwargs or use decorator argument
                # Plotting actual vs predicted for regressors
                plt.scatter(y_test, predictions, alpha=0.5)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title(f'{func.__name__}: Actual vs Predicted - {title_suffix}')
                plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line for perfect prediction
                plt.grid(True)
                plt.show()
                return X_test, y_test, predictions
            return wrapper
        return decorator_plot_results
    
    @_plot_classfier_with_title(custom_title="")
    def random_forest_fit(self, x_data, y_data, title_suffix=""):
        """_summary_

        Args:
            x_data (pd.DataFrame): feature_data
            y_data (pd.DataFrame | np.array): target_data

        Returns:
            _type_: _description_
        """
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=False)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        print(classification_report(y_test, predictions))
        return X_test, y_test, predictions

    @_plot_classfier_with_title(custom_title="")
    def decision_tree_fit(self, x_data, y_data, title_suffix=""):
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=False)
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        predictions = dt.predict(X_test)
        print(classification_report(y_test, predictions))
        return X_test, y_test, predictions

    @_plot_regressor_with_title(custom_title="")
    def svm_fit(self, x_data, y_data, title_suffix=""):
        # Always need to set shuffle as false since the data is time relavant
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=False)
        svm = LinearSVR()
        svm.fit(X_train, y_train)
        predictions = svm.predict(X_test)
        print("MSE:", mean_squared_error(y_test, predictions))
        return X_test, y_test, predictions

    @_plot_regressor_with_title(custom_title="")
    def lr_fit(self, x_data, y_data, title_suffix=""):
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=False)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        predictions = lr.predict(X_test)
        print("MSE:", mean_squared_error(y_test, predictions))
        return X_test, y_test, predictions

    @_plot_regressor_with_title(custom_title="")
    def xgboost_fit(self, x_data, y_data, title_suffix=""):
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=False)
        xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                     max_depth=5, alpha=10, n_estimators=100)
        xgb_model.fit(X_train, y_train)
        predictions = xgb_model.predict(X_test)
        print("MSE:", mean_squared_error(y_test, predictions))
        return X_test, y_test, predictions



    