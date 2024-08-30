

import pandas as pd
import numpy as np

def clean_data(data_path:str) -> pd.DataFrame:
    """Clean the data by removing any rows with missing values."""
    
    df = pd.read_csv('Multifunctional thiol data summary 2024_06_14_ML.csv')
    df['Gel'] = df['Gel'].replace(['yes', 'no'], [1, 0])
    
    return get_gel_df(df), get_nogel_df(df)

def get_gel_df(df: pd.DataFrame) -> pd.DataFrame:
    """Get the dataframe of gel samples."""
    gel_df = df[['Total_Conc_M', 'BDDA_mol%', 'Elp_mol%', 'Gel']]
    
    return gel_df

def get_nogel_df(df: pd.DataFrame) -> pd.DataFrame:
    """Get the dataframe of non-gel samples."""
    
    df = df.drop(columns=['SH_density_mmol_per_g'])
    nogel_df = df[df['Gel'] == False]
    nogel_df = nogel_df[nogel_df['Thiol_D'] != 'TBD']
    nogel_df = nogel_df[nogel_df['Thiol_Mw_kDa'] != 'TBD']
    nogel_df = nogel_df.dropna(axis=0) # Drop rows with nan
    nogel_df['Thiol_to_Parent_Mw_ratio'] = np.array(nogel_df['Thiol_Mw_kDa'], dtype=float) / np.array(nogel_df['Parent_Mw_kDa'], dtype=float)
    nogel_df['Thiol_to_Parent_D_ratio'] = np.array(nogel_df['Thiol_D'], dtype=float) / np.array(nogel_df['Parent_D'], dtype=float)
    
    
    nogel_df = nogel_df[['Total_Conc_M', 'BDDA_mol%', 'Elp_mol%', 'Thiol_to_Parent_Mw_ratio', 'Thiol_to_Parent_D_ratio']]
    return nogel_df

def merge_scope_with_data(scope_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    """Merge the scope dataframe with the data dataframe."""
    
    # Merge the data frames
    merged_df = pd.merge(scope_df, data_df, on=['Total_Conc_M', 'BDDA_mol%', 'Elp_mol%'], how='left')

    # Fill missing values with "PENDING"
    merged_df.fillna("PENDING", inplace=True)

    # Display the merged DataFrame
    # print(merged_df)
    
    return merged_df
        
    # # merged_df = pd.merge(scope_df, data_df, on=['Total_Conc_M', 'BDDA_mol%', 'Elp_mol%'], how='inner')
    # # df_edbo = pd.read_csv('my_optimization.csv')
    # merged_df = df_scope.merge(df[['Total_Conc_M', 'Elp_mol%', 'BDDA_mol%', 'Gel']],
    #                         on=['Total_Conc_M', 'Elp_mol%', 'BDDA_mol%'],
    #                         how='left',
    #                         suffixes=('', '_from_data'))
    # df_edbo['Gel'] = merged_df['Gel_from_data'].combine_first(df_edbo['Gel'])
    # df_edbo.loc[merged_df['Gel_from_data'].notna(), 'priority'] = -1
    # # df_edbo[df_edbo['Gel'] == 1e10]

    # return merged_df