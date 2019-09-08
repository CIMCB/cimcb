import numpy as np


def table_check(DataTable, PeakTable, print_statement=True):
    """Error checking for DataTable and PeakTable (used in load_dataXL).

    Parameters
    ----------
    DataTable: DataFrame
        Data sheet with the required columns.

    PeakTable: DataFrame
        Peak sheet with the required columns.

    print_statement: boolean (default True)
        If the error checks are successful and print_statement is True, the following is printed: "Data Table & Peak Table is suitable."
    """

    # Check DataTable for Idx, Class and SampleID
    data_columns = DataTable.columns.values

    if "Idx" not in data_columns:
        raise ValueError("Data Table does not contain the required 'Idx' column")
    if DataTable.Idx.isnull().values.any() == True:
        raise ValueError("Data Table Idx column cannot contain missing values")
    if len(np.unique(DataTable.Idx)) != len(DataTable.Idx):
        raise ValueError("Data Table Idx numbers are not unique. Please change")

    # Removed 'Class' as a required column
    # if "Class" not in data_columns:
    #     raise ValueError("Data Table does not contain the required 'Class' column")

    if "SampleID" not in data_columns:
        raise ValueError("Data Table does not contain the required 'SampleID' column")

    # Check PeakTable for Idx, Name, Label
    peak_columns = PeakTable.columns.values

    if "Idx" not in peak_columns:
        raise ValueError("Peak Table does not contain the required 'Idx' column")
    if PeakTable.Idx.isnull().values.any() == True:
        raise ValueError("Peak Table Idx column cannot contain missing values")
    if len(np.unique(PeakTable.Idx)) != len(PeakTable.Idx):
        raise ValueError("Peak Table Idx numbers are not unique. Please change")

    if "Name" not in peak_columns:
        raise ValueError("Peak Table does not contain the required 'Name' column")
    if PeakTable.Idx.isnull().values.any() == True:
        raise ValueError("Peak Table Name column cannot contain missing values")
    if len(np.unique(PeakTable.Idx)) != len(PeakTable.Idx):
        raise ValueError("Peak Table Name numbers are not unique. Please change")

    if "Label" not in peak_columns:
        raise ValueError("Data Table does not contain the required 'Label' column")

    # Check that Peak Names in PeakTable & DataTable match
    peak_list = PeakTable.Name
    data_columns = DataTable.columns.values
    temp = np.intersect1d(data_columns, peak_list)

    if len(temp) != len(peak_list):
        raise ValueError("The Peak Names in Data Table should exactly match the Peak Names in Peak Table. Remember that all Peak Names should be unique.")

    if print_statement is True:
        print("Data Table & Peak Table is suitable.")
