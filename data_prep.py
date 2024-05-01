import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

class DataPrep(object):
    def __init__(self):
        self.base_path = './data/'
        self.id_columns = ['date', 'id', 'ticker', 'sector', 'size_grp']
        self.ret_col = ['ret_exc_lead1m', 'ret_exc_lead1m_std']
        self.feature_col = None
    
    def _display_df_size(func):
        def wrapper(self, *args, **kwargs):
            df = func(self, *args, **kwargs)
            if isinstance(df, pd.DataFrame):
                print(f"DataFrame size: {df.shape}")  # Print the size of the DataFrame
            return df
        return wrapper
    
    @_display_df_size
    def data_initialization(self, return_df = False):
        """_summary_
        Read in relevant data
        """
        self.stock_price_df = pd.read_pickle(f'{self.base_path}crsp_ticker_price.pickle')[['PERMNO','TICKER']]
        self.stock_data = pd.read_pickle(os.path.join('./data/', 'usa.pkl')) 
        self.unique_ticker_sector_mapper = pd.read_csv(f'{self.base_path}ticker2sector.csv')[['Symbol', 'Sector']].to_dict('records')
        
        print('data initialized')
        if return_df:
            return self.stock_data
    
    @_display_df_size
    def data_construction(self, return_df = False):
        """_summary_
        Since the data I expected was not in one place, therefore I had to improvise and remap relevant information. Including dropping some records that I don't recognize
        """
        # Check if input is valid 
        assert isinstance(self.stock_price_df, pd.DataFrame)
        assert isinstance(self.stock_data, pd.DataFrame)
        assert isinstance(self.unique_ticker_sector_mapper, list)
        if any([self.stock_price_df.empty, self.stock_data.empty, not self.unique_ticker_sector_mapper]):
            raise ValueError('Empty Input Data')
        
        # Here we try to find interpolate the sector information
        unique_permno_ticker = self.stock_price_df.drop_duplicates()
        # Since a lot the ticker names have changed, we will map sector information only using the lastest tciker
        # This is not a perfect method, but it works for now
        unique_permno_ticker_last = unique_permno_ticker.groupby('PERMNO').last()
        unique_permno_ticker_last_mapper = unique_permno_ticker_last.reset_index().to_dict('records')
        unique_permno_ticker_last_mapper_dict = {}
        for record in unique_permno_ticker_last_mapper:
            unique_permno_ticker_last_mapper_dict[record['PERMNO']] = record['TICKER']
        # Release Cache
        del self.stock_price_df
        
        # Now we map the CRSP symbol data into the factor data
        self.stock_data['ticker'] = self.stock_data['id'].map(unique_permno_ticker_last_mapper_dict)
        # Drop missing ticker data because we will not be able to identify sector from missing ticker data
        self.stock_data = self.stock_data.dropna(subset=['ticker'])
        # Here we will map the sector information into our factor data
        unique_ticker_sector_mapper_dict = {}
        for record in self.unique_ticker_sector_mapper:
            unique_ticker_sector_mapper_dict[record['Symbol']] = record['Sector']
        self.stock_data['sector'] = self.stock_data['ticker'].map(unique_ticker_sector_mapper_dict)
        # Release Cache
        del self.unique_ticker_sector_mapper
        # Drop missing values because we won't be able to analyze missing sector data
        self.stock_data = self.stock_data.dropna(subset=['sector'])
        
        # Normalize all numerical data
        scaler = StandardScaler()
        self.stock_data[[col for col in self.stock_data.columns if col not in (self.id_columns+self.ret_col)]]  = scaler.fit_transform(self.stock_data[[col for col in self.stock_data.columns if col not in (self.id_columns+self.ret_col)]])
        self.stock_data['ret_exc_lead1m_std'] = pd.DataFrame(scaler.fit_transform(self.stock_data[['ret_exc_lead1m']]))
        
        # Rearrange data
        self.feature_col = [item for item in self.stock_data.columns if item not in (self.id_columns+self.ret_col)]

        self.stock_data = self.stock_data[self.id_columns+self.feature_col+self.ret_col]
        
        print('data construction complete')
        if return_df:
            return self.stock_data
    
    
    def data_slicing(self):
        # Stock id data
        id_df = self.stock_data[self.id_columns]
        # Stock Features
        feature_df = self.stock_data[[col for col in self.stock_data.columns if col not in (self.id_columns+self.ret_col)]]
        # Stock Return
        return_df = self.stock_data[self.ret_col]
        return id_df, feature_df, return_df
        
    
        
if __name__ == "__main__":
    data_obj = DataPrep()
    data_obj.data_initialization()
    data_obj.data_construction()
    # data initialized
    # DataFrame size: (4135225, 135)
    # data construction complete
    # DataFrame size: (987322, 137)
    # Notes: the data was only 1/4 of the original size