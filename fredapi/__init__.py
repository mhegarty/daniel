import pandas as pd
import requests
from typing import Iterable


class FredAPI:
    def __init__(self, api_key: str) -> None:
        """
        Initialize the FredAPI with the given API key.

        See https://fred.stlouisfed.org/docs/api/api_key.html for more information about API keys.

        :param api_key: Your FRED API key.
        """
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
    
    def search_series(self, search_text: str) -> pd.DataFrame:
        """
        Search for series by keywords.

        :param search_text: The text to search for.
        :return: A dataframe containing the search results.

        Example:
        >>> api = FredAPI(api_key="your_api_key")
        >>> df = api.search_series(search_text="GDP")
        >>> df.head()
        """
        url = f"{self.base_url}/series/search?"
        params = {
            'api_key': self.api_key,
            'file_type': 'json',
            'search_text': search_text
        }
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Create a dataframe with search results
        data = response.json()
        df = pd.DataFrame(data['seriess'])

        # Add metadata to the dataframe
        metadata = {k: data[k] for k in data if k!='seriess'}
        for key, value in metadata.items():
            df.attrs[key] = value

        return df

    def get_panel(self, series_id: str, observation_dates: Iterable, window: int = 24) -> pd.DataFrame:
        """
        Get a panel of data for a given series ID.

        :param series_id: The series ID to retrieve.
        :param observation_dates: A list of dates to retrieve data for.
        :param window: The number of periods back to retrieve for each observation date.
        
        :return: A dataframe containing the panel data.

        Example:
        >>> api = FredAPI(api_key="your_api_key")
        >>> observation_dates = pd.date_range(start="2020-01-01", end="2025-01-01", freq='M')
        >>> df = api.get_panel(series_id="RSXFSN", observation_dates=observation_dates)
        >>> df.head()
        """
        url = f"{self.base_url}/series/observations?"
        params = {
            'api_key': self.api_key,
            'file_type': 'json',
            'series_id': series_id,
            'realtime_start': observation_dates.min().strftime('%Y-%m-%d'),
            'realtime_end': observation_dates.max().strftime('%Y-%m-%d'),
        }
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        metadata = {k: data[k] for k in data if k!='observations'}

        data = pd.DataFrame(data['observations'])

        # Paginate if needed (the rest endpoint has a limit of 100,000 records per call)
        while metadata['count'] > len(data):
            params['offset'] = len(data)
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = pd.concat([data, pd.DataFrame(response.json()['observations'])])
        
        # Convert dates to datetime
        for col in ['realtime_start', 'realtime_end', 'date']:
            data[col] = pd.to_datetime(data[col])

        # Create a panel of data
        # Create a dataframe with an index of observation dates
        observation_df = pd.DataFrame(observation_dates, columns=['observation_date'])
        
        # Merge on the product of observation dates and the original dataframe
        df = observation_df.merge(data, how='cross')

        # Filter where the observation date is in between the realtime dates
        df = df[(df['observation_date'] >= df['realtime_start']) & 
                    (df['observation_date'] <= df['realtime_end'])]

        # Apply the window filter
        if window:
            df = df.groupby('observation_date') \
                    .apply(lambda x: x.nlargest(window, 'date')) \
                    .reset_index(drop=True)
        
        # Add a column to identify the number of periods back
        df['periods_back'] = df.sort_values('date', ascending=False) \
                            .groupby('observation_date')['date'] \
                            .transform('cumcount') + 1
        
        # Pivot the dataframe
        df = df.pivot(index='observation_date', columns='periods_back', values=['value', 'date'])

        # Add metadata to the dataframe
        for key, value in metadata.items():
            df.attrs[key] = value
        
        return df.stack(future_stack=True)
