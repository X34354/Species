#ffile to create a database in which we can know which files are going to run for train test and val
import pandas as pd
import numpy as np
import os

def filter_df(df, df_species, duration):
    """
    Filter the DataFrame 'df' based on the specified species and duration.

    Args:
        df (pandas.DataFrame): DataFrame to be filtered.
        df_species (pandas.DataFrame): DataFrame containing the species IDs to filter.
        duration (int): Duration value to filter.

    Returns:
        pandas.DataFrame: Filtered DataFrame containing only the matching rows.

    """

    # Filter the DataFrame based on the species IDs
    df_filter = df[df['SpeciesID'].isin(df_species['SpeciesID'])].copy()

    # Filter the DataFrame based on the duration
    df_filter = df_filter[df_filter['Duration'] == duration]

    return df_filter


def split_data_by_group(df_files, df_groups, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    """
    Split the data in the 'df_files' dataframe into 'train', 'test', and 'val' sets based on the groups specified in
    the 'df_groups' dataframe.

    Args:
        df_files (pandas.DataFrame): DataFrame containing the filenames and groups.
        df_groups (pandas.DataFrame): DataFrame containing the groups.
        train_ratio (float, optional): Proportion of data to allocate to the 'train' set. Defaults to 0.7.
        test_ratio (float, optional): Proportion of data to allocate to the 'test' set. Defaults to 0.2.
        val_ratio (float, optional): Proportion of data to allocate to the 'val' set. Defaults to 0.1.

    Returns:
        pandas.DataFrame: The 'df_files' dataframe with an additional column 'Ml' indicating the set for each file.

    """

    # Create a column 'Ml' initially with 'train' values in the dataframe
    df_files['Ml'] = 'train'

    # Iterate over the unique groups and generate masks and assignments
    for group in df_groups['SpeciesID']:
        # Get the mask for the current group
        group_mask = df_files['SpeciesID'] == group

        # Calculate the sizes of the data sets for the current group
        total_rows = group_mask.sum()
        train_size = int(total_rows * train_ratio)
        test_size = int(total_rows * test_ratio)
        val_size = total_rows - train_size - test_size

        # Get the indices of the rows for each data set within the group
        train_indices = np.random.choice(df_files[group_mask].index, size=train_size, replace=False)
        remaining_indices = np.setdiff1d(df_files[group_mask].index, train_indices)
        test_indices = np.random.choice(remaining_indices, size=test_size, replace=False)
        val_indices = np.setdiff1d(remaining_indices, test_indices)

        # Assign the values to the 'Ml' column based on the indices and mask
        df_files.loc[train_indices, 'Ml'] = 'train'
        df_files.loc[test_indices, 'Ml'] = 'test'
        df_files.loc[val_indices, 'Ml'] = 'val'

    return df_files

def test_split_data_by_group(df_files,df_groups):

    # Perform the data split
    df_result = split_data_by_group(df_files, df_groups, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1)

    # Check that the 'Ml' column is added and has the correct values
    assert 'Ml' in df_result.columns
    assert df_result['Ml'].isin(['train', 'test', 'val']).all()

    # Check that the sum of rows in each set matches the expected proportions
    train_rows = (df_result['Ml'] == 'train').sum()
    test_rows = (df_result['Ml'] == 'test').sum()
    val_rows = (df_result['Ml'] == 'val').sum()
    total_rows = len(df_result)

    print(f"Train rows: {train_rows}")
    print(f"Test rows: {test_rows}")
    print(f"Val rows: {val_rows}")
    print(f"Total rows: {total_rows}")

    assert np.isclose(train_rows / total_rows, 0.7, atol=0.01), f"Train rows proportion doesn't match expected value."
    assert np.isclose(test_rows / total_rows, 0.2, atol=0.01), f"Test rows proportion doesn't match expected value."
    assert np.isclose(val_rows / total_rows, 0.1, atol=0.01), f"Val rows proportion doesn't match expected value."

    print("All tests passed!")

def filter_existing_videos(df_arc, name ):
    # Create a mask for existing videos
    mask = df_arc[str(name)].apply(lambda x: os.path.exists(f'G:/videos/{x}'))

    # Filter the dataframe
    filtered_df = df_arc[mask]

    return filtered_df

if __name__ == '__main__' :
    #read data base
    df = pd.read_excel('data/Base.xlsx')
    #the names of the species we are interested in
    df_specie = pd.read_excel('data/Especies Red.xlsx')
    # Create a new DataFrame with the filtered records
    filtered_df = filter_df(df,df_specie,20)

    filtered_df_exis = filter_existing_videos(filtered_df, 'File' )
    # Run the unit tests
    test_split_data_by_group(filtered_df_exis, df_specie)

    df_split  = split_data_by_group(filtered_df_exis, df_specie, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1)

    df_split.to_csv('data/df_filter.csv')