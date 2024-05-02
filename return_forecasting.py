from feature_predictions import FeaturePredictionMethods
import pandas as pd

class ReturnForecast(FeaturePredictionMethods):
    def __init__(self):
        super().__init__()
        
    def daily_level_portfolio_return_construction(self, y_test, predictions, fee_rate=0, predictor_type='discrete'):
        """_summary_

        Args:
            df (_type_): this is the predictions
            sector (_type_): _description_
        """
        signal_df = self.stock_data[self.stock_data.index.isin(y_test.index)].reset_index()
        signal_df = pd.concat([signal_df, pd.Series(predictions, name='predictions')], axis=1)
        if predictor_type == 'discrete':
            signal_df['signal'] = signal_df['predictions'].map({1:-1, 2:0, 3:1})
        else:
            signal_df['predictions'] = signal_df['predictions'].fillna(0)
            print(signal_df.isnull().any().any())
            signal_df['signal'] = pd.cut(signal_df['predictions'], [-99999, -0.01, 0.01, 99999], labels=[-1, 0, 1]).astype(int)
        signal_df['fee'] = (fee_rate * signal_df['signal'].diff().abs()).fillna(0)
        signal_df['signal_ret'] = signal_df['signal'] * signal_df['ret_exc_lead1m'] - signal_df['fee'] 
        signal_grouped_df = pd.DataFrame((signal_df.groupby('date')['signal_ret'].mean()+1).cumprod())
        signal_grouped_df.index = pd.to_datetime(signal_grouped_df.index, format='%Y%m%d')
        return signal_df, signal_grouped_df

    