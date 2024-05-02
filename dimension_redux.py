from data_prep import DataPrep

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

class DimensionRedux(DataPrep):
    def __init__(self):
        super().__init__()
    
    @DataPrep._display_df_size   
    def data_clean_na(self, df):
        df = df.dropna(axis=0)
        return df
        
    def generate_ret_class(self, df, n=2):
        class_bins = [-9999999]+[df['ret_exc_lead1m'].quantile(i) for i in np.arange(0,1,1/n)]+[9999999999]
        df['ret_class'] = pd.cut(df['ret_exc_lead1m'], class_bins, labels=[i+1 for i in range(n+1)]).astype(int)
        return df
        
    def pca_fit(self):
        """_summary_
        Reducing the dimensionality of the dataset by identifying the principal components that capture the most variance in the data.
        """
        # Remove missing values
        stock_data = self.data_clean_na(df=self.stock_data).copy()
        # Set Return class
        stock_data = self.generate_ret_class(df=stock_data).copy()
        
        # Check for missing values 
        if stock_data.isnull().any().any():
            raise ValueError('Missing Values are not allowed in PCA Method')
        
        # Model Fitting and generate compressed components
        pca = PCA()   
        stock_df_pca_model_obj = pca.fit(stock_data[self.feature_col])
        
        # Construct new reduced df stat 
        pca_index_val = np.arange(0, len(stock_df_pca_model_obj.explained_variance_ratio_), 1)
        pca_index_rename = [f'comp_{i}' for i in pca_index_val]
        pca_var = stock_df_pca_model_obj.explained_variance_ratio_
        pca_cumu_var = stock_df_pca_model_obj.explained_variance_ratio_.cumsum()
        pca_result_df = pd.DataFrame([pca_index_val, pca_index_rename, pca_var, pca_cumu_var]).T
        pca_result_df.columns = ['index_values', 'renamed_columns', 'expl_var', 'expl_var_cumu']
        
        # Find out when did we reach 80% explanation of variance
        pca_result_df_80 = pca_result_df[pca_result_df.expl_var_cumu<=0.8].copy()
        stock_df_trans = pd.DataFrame(data=pca.transform(stock_data[self.feature_col]))
        stock_df_trans_80 = stock_df_trans[pca_result_df_80.index_values]
        stock_df_trans_80.columns = pca_result_df_80.renamed_columns
        stock_df_trans_80 = pd.concat([stock_data[self.id_columns].reset_index(), stock_df_trans_80.reset_index(), stock_data[self.ret_col+['ret_class']].reset_index()], axis=1)
        
        # Some Visualizations
        # Visualize Variance Explained
        plt.figure(figsize=(10,6))
        plt.scatter(x=[i+1 for i in range(len(stock_df_pca_model_obj.explained_variance_ratio_))],
                    y=stock_df_pca_model_obj.explained_variance_ratio_.cumsum(),
                s=200, alpha=0.75,c='orange',edgecolor='k')
        plt.grid(True)
        plt.title("Explained variance ratio of the \nfitted principal component vector\n",fontsize=18)
        plt.xlabel("Principal components",fontsize=15)
        plt.xticks([i+1 for i in range(len(stock_df_pca_model_obj.explained_variance_ratio_))],fontsize=15)

        # Setting x-ticks interval
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(8))  # Set interval of 1 to ensure each principal component is marked

        plt.yticks(fontsize=15)
        plt.ylabel("Cumu Explained variance ratio",fontsize=15)
        plt.show()
        
        # Assuming stock_df_trans_80 is already defined and contains 'comp_0' and 'comp_1'

        # Calculate IQR for 'comp_0'
        q1_comp0 = stock_df_trans_80['comp_0'].quantile(0.05)
        q3_comp0 = stock_df_trans_80['comp_0'].quantile(0.95)
        iqr_comp0 = q3_comp0 - q1_comp0

        # Calculate IQR for 'comp_1'
        q1_comp1 = stock_df_trans_80['comp_1'].quantile(0.05)
        q3_comp1 = stock_df_trans_80['comp_1'].quantile(0.95)
        iqr_comp1 = q3_comp1 - q1_comp1

        # Define bounds for outliers
        lower_bound_comp0 = q1_comp0 - 1.5 * iqr_comp0
        upper_bound_comp0 = q3_comp0 + 1.5 * iqr_comp0
        lower_bound_comp1 = q1_comp1 - 1.5 * iqr_comp1
        upper_bound_comp1 = q3_comp1 + 1.5 * iqr_comp1

        # Filter out outliers
        filtered_df = stock_df_trans_80[
    (stock_df_trans_80['comp_0'] >= lower_bound_comp0) & (stock_df_trans_80['comp_0'] <= upper_bound_comp0) &
    (stock_df_trans_80['comp_1'] >= lower_bound_comp1) & (stock_df_trans_80['comp_1'] <= upper_bound_comp1)
]
        
        # Visualized component 0 and component 1 without outliers
        plt.scatter(filtered_df['comp_0'], filtered_df['comp_1'], c=filtered_df['ret_class'], edgecolors='k', alpha=0.75, s=150)
        plt.grid(True)
        plt.title("Class separation using first two principal components\n(no outliers)", fontsize=20)
        plt.xlabel("Principal component-1", fontsize=15)
        plt.ylabel("Principal component-2", fontsize=15)
        plt.show()
        
        return stock_df_trans_80
    
    def lda_fit(self):
        """_summary_
        Perform LDA to reduce dimensions focusing on maximizing class separation.
        """
        
        # Remove missing values
        stock_data = self.data_clean_na(df=self.stock_data).copy()
        # Set Return class
        stock_data = self.generate_ret_class(df=stock_data).copy()
        
        # Check for missing values 
        if stock_data.isnull().any().any():
            raise ValueError('Missing Values are not allowed in LDA Method')
        
        # Fit LDA model
        lda_act_model = LDA()
        # define model evaluation method
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(lda_act_model, stock_data[self.feature_col], stock_data['ret_class'], scoring='accuracy', cv=cv, n_jobs=-1)
        print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores))) 
        
        stock_data_lda = lda_act_model.fit_transform(stock_data[self.feature_col], stock_data['ret_class'])
        
        # Create a DataFrame for the LDA components
        lda_columns = ['lda_comp_{}'.format(i) for i in range(stock_data_lda.shape[1])]
        stock_data_lda_df = pd.DataFrame(stock_data_lda, columns=lda_columns)
        
        # Merge with key columns and target for further analysis
        stock_data_lda_df = pd.concat([stock_data[self.id_columns].reset_index(drop=True),
                                       stock_data_lda_df,
                                       stock_data[self.ret_col+['ret_class']].reset_index(drop=True)], axis=1)
        

        # Visualization of the LDA components
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(stock_data_lda_df['lda_comp_0'], stock_data_lda_df['lda_comp_1'],
                              c=stock_data_lda_df['ret_class'], cmap='viridis', edgecolor='k', alpha=0.75, s=150)
        plt.colorbar(scatter)
        plt.grid(True)
        plt.title("Class separation using first two LDA components\n", fontsize=20)
        plt.xlabel("LDA Component 1", fontsize=15)
        plt.ylabel("LDA Component 2", fontsize=15)
        plt.show()
        return stock_data_lda_df
    
    def svd_fit(self):
        """_summary_
        Perform SVD to identify directions with maximum variance in the dataset.
        """
        # Remove missing values
        stock_data = self.data_clean_na(df=self.stock_data).copy()
        # Set Return class
        stock_data = self.generate_ret_class(df=stock_data).copy()
        
        # Check for missing values 
        if stock_data.isnull().any().any():
            raise ValueError('Missing Values are not allowed in SVD Method')
        
        # Choose the number of components, for example:
        n_components = min(stock_data[self.feature_col].shape) - 1  # Or some other logic
        
        # Model Fitting and generate compressed components
        svd = TruncatedSVD(n_components=n_components)   
        stock_df_svd_model_obj = svd.fit(stock_data[self.feature_col])
        
        # Construct new reduced df stat 
        svd_index_val = np.arange(0, len(stock_df_svd_model_obj.explained_variance_ratio_), 1)
        svd_index_rename = [f'comp_{i}' for i in svd_index_val]
        svd_var = stock_df_svd_model_obj.explained_variance_ratio_
        svd_cumu_var = stock_df_svd_model_obj.explained_variance_ratio_.cumsum()
        svd_result_df = pd.DataFrame([svd_index_val, svd_index_rename, svd_var, svd_cumu_var]).T
        svd_result_df.columns = ['index_values', 'renamed_columns', 'expl_var', 'expl_var_cumu']
        
        # Find out when did we reach 80% explanation of variance
        svd_result_df_80 = svd_result_df[svd_result_df.expl_var_cumu<=0.8].copy()
        stock_df_trans = pd.DataFrame(data=svd.transform(stock_data[self.feature_col]))
        stock_df_trans_80 = stock_df_trans[svd_result_df_80.index_values]
        stock_df_trans_80.columns = svd_result_df_80.renamed_columns
        stock_df_trans_80 = pd.concat([stock_data[self.id_columns].reset_index(), stock_df_trans_80.reset_index(), stock_data[self.ret_col+['ret_class']].reset_index()], axis=1)
        
        
        # Visualization of the singular values
        plt.figure(figsize=(10, 6))
        plt.bar(range(stock_df_svd_model_obj.n_components), stock_df_svd_model_obj.explained_variance_ratio_.cumsum())
        plt.xlabel('SVD Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by SVD Components')
        plt.show()
        
        # Calculate IQR for 'comp_0'
        q1_comp0 = stock_df_trans_80['comp_0'].quantile(0.05)
        q3_comp0 = stock_df_trans_80['comp_0'].quantile(0.95)
        iqr_comp0 = q3_comp0 - q1_comp0

        # Calculate IQR for 'comp_1'
        q1_comp1 = stock_df_trans_80['comp_1'].quantile(0.05)
        q3_comp1 = stock_df_trans_80['comp_1'].quantile(0.95)
        iqr_comp1 = q3_comp1 - q1_comp1

        # Define bounds for outliers
        lower_bound_comp0 = q1_comp0 - 1.5 * iqr_comp0
        upper_bound_comp0 = q3_comp0 + 1.5 * iqr_comp0
        lower_bound_comp1 = q1_comp1 - 1.5 * iqr_comp1
        upper_bound_comp1 = q3_comp1 + 1.5 * iqr_comp1

        # Filter out outliers
        filtered_df = stock_df_trans_80[
    (stock_df_trans_80['comp_0'] >= lower_bound_comp0) & (stock_df_trans_80['comp_0'] <= upper_bound_comp0) &
    (stock_df_trans_80['comp_1'] >= lower_bound_comp1) & (stock_df_trans_80['comp_1'] <= upper_bound_comp1)
]
        
        # Visualized component 0 and component 1 without outliers
        plt.scatter(filtered_df['comp_0'], filtered_df['comp_1'], c=filtered_df['ret_class'], edgecolors='k', alpha=0.75, s=150)
        plt.grid(True)
        plt.title("Class separation using first two principal components\n(no outliers)", fontsize=20)
        plt.xlabel("Principal component-1", fontsize=15)
        plt.ylabel("Principal component-2", fontsize=15)
        plt.show()
        
        return stock_df_trans_80
    
    def fa_fit(self):
        """_summary_
        Reducing the dimensionality of the dataset by identifying latent factors that explain the most variance in the data."""
        # Remove missing values
        stock_data = self.data_clean_na(df=self.stock_data).copy()
        # Set Return class
        stock_data = self.generate_ret_class(df=stock_data).copy()

        # Check for missing values
        if stock_data.isnull().any().any():
            raise ValueError('Missing Values are not allowed in FA Method')

        # Model Fitting and generate compressed components
        fa = FactorAnalysis()
        fa.fit(stock_data[self.feature_col])
        
        # Transform data using the FA model
        stock_data_transformed = fa.transform(stock_data[self.feature_col])

        # Construct new reduced DataFrame for statistics
        fa_index_val = np.arange(0, len(fa.components_), 1)
        fa_index_rename = [f'factor_{i}' for i in fa_index_val]
        fa_var = np.var(stock_data_transformed, axis=0)
        fa_cumu_var = np.cumsum(fa_var)
  
        # Visualization of Explained Variance
        plt.figure(figsize=(10, 6))
        plt.bar(fa_index_val, fa_var, color='blue', label='Individual Explained Variance')
        plt.plot(fa_index_val, fa_cumu_var, color='red', marker='o', linestyle='-', label='Cumulative Explained Variance')
        plt.xlabel('Factors')
        plt.ylabel('Explained Variance')
        plt.title('Explained Variance by Factor')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Scatter plot for the first two factors
        plt.figure(figsize=(10, 6))
        plt.scatter(stock_data_transformed[:, 0], stock_data_transformed[:, 1], c=stock_data['ret_class'], cmap='viridis', edgecolor='k', alpha=0.75)
        plt.colorbar(label='Return Class')
        plt.xlabel('Factor 0')
        plt.ylabel('Factor 1')
        plt.title('Scatter Plot of First Two Factors')
        plt.grid(True)
        plt.show()

        # Create a DataFrame for the transformed data and merge with original identifiers and return classes
        stock_data_factors_df = pd.DataFrame(stock_data_transformed, columns=fa_index_rename)
        stock_data_factors_df = pd.concat([stock_data[self.id_columns].reset_index(drop=True),
                                           stock_data_factors_df,
                                           stock_data[self.ret_col + ['ret_class']].reset_index(drop=True)], axis=1)

        return stock_data_factors_df
        
        
if __name__ == "__main__":
    dr_obj = DimensionRedux()
    dr_obj.data_initialization()
    dr_obj.data_construction()
    dr_obj.fa_fit()
    # data initialized
    # DataFrame size: (4135225, 135)
    # data construction complete
    # DataFrame size: (987322, 137)
    # Notes: the data was only 1/4 of the original size