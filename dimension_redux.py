from data_prep import DataPrep

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

class DimensionRedux(DataPrep):
    def __init__(self):
        super().__init__()
        
    def generate_ret_class(self):
        class_bins = [-9999999]+[self.stock_data['ret_exc_lead1m'].quantile(i) for i in np.arange(0,1,0.2)]+[9999999999]
        self.stock_data['ret_class'] = pd.cut(self.stock_data['ret_exc_lead1m'], class_bins, labels=[1,2,3,4,5,6]).astype(int)
        
    def pca_fit(self):
        """_summary_
        """
        # Remove missing values
        self.data_clean_na()
        self.generate_ret_class()
        # TODO: Check for missing values 
        # Model Fitting and generate compressed components
        pca = PCA()   
        stock_df_pca_model_obj = pca.fit(self.stock_data[self.feature_col])
        pca_index_val = np.arange(0, len(stock_df_pca_model_obj.explained_variance_ratio_), 1)
        pca_index_rename = [f'comp_{i}' for i in pca_index_val]
        pca_var = stock_df_pca_model_obj.explained_variance_ratio_
        pca_cumu_var = stock_df_pca_model_obj.explained_variance_ratio_.cumsum()
        pca_result_df = pd.DataFrame([pca_index_val, pca_index_rename, pca_var, pca_cumu_var]).T
        pca_result_df.columns = ['index_values', 'renamed_columns', 'expl_var', 'expl_var_cumu']
        # Find out when did we reach 80% explanation of variance
        pca_result_df_80 = pca_result_df[pca_result_df.expl_var_cumu<=0.8].copy()
        stock_df_trans = pd.DataFrame(data=pca.transform(self.stock_data[self.feature_col]))
        stock_df_trans_80 = stock_df_trans[pca_result_df_80.index_values]
        stock_df_trans_80.columns = pca_result_df_80.renamed_columns
        stock_df_trans_80 = pd.concat([self.stock_data[self.id_columns].reset_index(), stock_df_trans_80.reset_index(), self.stock_data[self.ret_col].reset_index()], axis=1)
        
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
        
        # Visualized component 0 and component 1
        plt.scatter(stock_df_trans_80['comp_0'],stock_df_trans_80['comp_1'],c=stock_df_trans_80['ret_class'],edgecolors='k',alpha=0.75,s=150)
        plt.grid(True)
        plt.title("Class separation using first two principal components\n",fontsize=20)
        plt.xlabel("Principal component-1",fontsize=15)
        plt.ylabel("Principal component-2",fontsize=15)
        plt.show()
        
        return stock_df_trans_80
    
    def lda_fit(self):
        """_summary_
        Perform LDA to reduce dimensions focusing on maximizing class separation.
        """
        
        # Remove missing values (if any)
        self.data_clean_na()
        self.generate_ret_class()
        
        # Determine how many components to keep
        # Assuming X is your feature matrix and y are your labels
        max_components = min(len(np.unique(self.stock_data['ret_class'])) - 1, self.stock_data[self.feature_col].shape[1])
        scores = []
        for n in range(1, max_components + 1):
            lda = LDA(n_components=n)
            # pipeline = make_pipeline(lda, )
            cv_scores = cross_val_score(lda, self.stock_data[self.feature_col], self.stock_data['ret_class'], cv=StratifiedKFold(5), scoring='accuracy')
            print(cv_scores)
            scores.append(np.mean(cv_scores))

        plt.plot(range(1, max_components + 1), scores)
        plt.xlabel('Number of Components')
        plt.ylabel('CV Accuracy')
        plt.show()

        best_n_components = np.argmax(scores) + 1
        print(f"Best number of components: {best_n_components}")
        
        # Fit LDA model
        lda_act_model = LDA(n_components=best_n_components)  # Using 2 components as an example
        stock_data_lda = lda_act_model.fit_transform(self.stock_data[self.feature_col], self.stock_data['ret_class'])
        
        # Create a DataFrame for the LDA components
        lda_columns = ['lda_comp_{}'.format(i) for i in range(stock_data_lda.shape[1])]
        stock_data_lda_df = pd.DataFrame(stock_data_lda, columns=lda_columns)
        
        # Merge with key columns and target for further analysis
        stock_data_lda_df = pd.concat([self.stock_data[self.id_columns].reset_index(drop=True),
                                       stock_data_lda_df,
                                       self.stock_data[self.ret_col+['ret_class']].reset_index(drop=True)], axis=1)
        

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
        """
        pass
    
    def fa_fit(self):
        """_summary_
        """
        
if __name__ == "__main__":
    dr_obj = DimensionRedux()
    dr_obj.data_initialization()
    dr_obj.data_construction()
    dr_obj.lda_fit()
    # data initialized
    # DataFrame size: (4135225, 135)
    # data construction complete
    # DataFrame size: (987322, 137)
    # Notes: the data was only 1/4 of the original size