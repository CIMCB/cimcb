import pandas as pd
import numpy as np
from os import path
from .table_check import table_check


def load_dataCSV(DataSheet, PeakSheet):
    """Loads and validates the DataFile and PeakFile from csv files.


    Parameters
    ----------
    DataSheet : string
        The name of the csv file (.csv file) that contains the 'Data'. Note, the data sheet must contain an 'Idx' and 'SampleID'column. e.g. 'datasheetxxx1.csv' or '/homedir/datasheetxxx1.csv'

    PeakSheet : string
        The name of the csv file (.csv file) that contains the 'Peak'. Note, the peak sheet must contain an 'Idx', 'Name', and 'Label' column. e.g. 'peaksheetxxx1.csv' or 'peaksheetxxx1.csv'

    Returns
    -------
    DataTable: DataFrame
        Data sheet from the csv file.

    PeakTable: DataFrame
        Peak sheet from the csv file.
    """

    # Check Datasheet exists
    if path.isfile(DataSheet) is False:
        raise ValueError("{} does not exist.".format(filename))

    if not DataSheet.endswith(".csv"):
        raise ValueError("{} should be a .csv file.".format(filename))

    # Check PeakSheet exists
    if path.isfile(PeakSheet) is False:
        raise ValueError("{} does not exist.".format(filename))

    if not PeakSheet.endswith(".csv"):
        raise ValueError("{} should be a .csv file.".format(filename))

    # LOAD PEAK DATA
    print("Loadings PeakFile: {}".format(PeakSheet))
    PeakTable = pd.read_csv(PeakSheet)

    # LOAD DATA TABLE
    print("Loadings DataFile: {}".format(DataSheet))
    DataTable = pd.read_csv(DataSheet)

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
