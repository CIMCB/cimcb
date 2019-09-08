import pandas as pd
import numpy as np
from os import path
from .table_check import table_check


def load_dataXL(filename, DataSheet, PeakSheet):
    """Loads and validates the DataFile and PeakFile from an excel file.


    Parameters
    ----------
    file : string
        The name of the excel file (.xlsx file) e.g. 'projectxxx1.xlsx'.  Note, it can include the directory e.g. '/homedir/projectxxx1.xlsx'

    DataSheet : string
        The name of the data sheet in the file e.g. 'Data'. Note, the data sheet must contain an 'Idx' and 'SampleID' column.

    PeakSheet : string
        The name of the peak sheet in the file e.g. 'Pata'. Note, the peak sheet must contain an 'Idx', 'Name', and 'Label' column.

    Returns
    -------
    DataTable: DataFrame
        Data sheet from the excel file.

    PeakTable: DataFrame
        Peak sheet from the excel file.
    """

    if path.isfile(filename) is False:
        raise ValueError("{} does not exist.".format(filename))

    if not filename.endswith(".xlsx"):
        raise ValueError("{} should be a .xlsx file.".format(filename))

    # LOAD PEAK DATA
    print("Loadings PeakFile: {}".format(PeakSheet))
    PeakTable = pd.read_excel(filename, sheet_name=PeakSheet)

    # LOAD DATA TABLE
    print("Loadings DataFile: {}".format(DataSheet))
    DataTable = pd.read_excel(filename, sheet_name=DataSheet)

    # Replace with nans
    DataTable = DataTable.replace(-99, np.nan)
    DataTable = DataTable.replace(".", np.nan)
    DataTable = DataTable.replace(" ", np.nan)

    # Error checks
    table_check(DataTable, PeakTable, print_statement=True)

    # Make the Idx column start from 1
    DataTable.index = np.arange(1, len(DataTable) + 1)
    PeakTable.index = np.arange(1, len(PeakTable) + 1)

    print("TOTAL SAMPLES: {} TOTAL PEAKS: {}".format(len(DataTable), len(PeakTable)))
    print("Done!")
    return DataTable, PeakTable
