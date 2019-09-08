import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests
from .table_check import table_check


def univariate_2class(DataTable, PeakTable, group, posclass, parametric=True, seed=None):
    """Creates a table of univariate statistics (2 class).

    Parameters
    ----------
    DataTable: DataFrame
        Data sheet with the required columns.

    PeakTable: DataFrame
        Peak sheet with the required columns.

    group: string
        Name of the column in the DataTable that contains the Class data.

    posclass: number or string
        Name of the positive class in the group column.

    parametric: boolean (default True)
        If parametric is False, Median 95%CI and Mann-Whitney U test is calculated instead of Mean 95%CI and TTest.

    seed: integer or None (default None)
        Used to seed the generator for the Median 95% CI bootstrap (resample with replacement). Note, if parametric=True the seed is irrelevant as Mean 95% CI is calculated parametrically.

    Returns
    -------
    StatsTable: DataFrame
        Table that contains multiple univariate statistics (2 class).
    """

    # Error checks
    table_check(DataTable, PeakTable, print_statement=False)
    if group not in DataTable:
        raise ValueError("Column '{}' does not exist in DataTable".format(group))
    if posclass not in DataTable[group].unique():
        raise ValueError("Positive class was not found in '{}' column.".format(group))
    if len(DataTable[group].unique()) != 2:
        raise ValueError("Column '{}' should have exactly 2 groups".format(group))

    # Get x0, x1
    peaklist = PeakTable["Name"]
    x = DataTable[peaklist]
    y = DataTable[group]
    x1 = x[y == posclass]  # x1 refers to posclass
    x0 = x[y != posclass]

    # Create stats table (include Idx, Name and Label)
    StatsTable = pd.DataFrame()
    StatsTable["Idx"] = PeakTable["Idx"]
    StatsTable["Name"] = PeakTable["Name"]
    StatsTable["Label"] = PeakTable["Label"]

    if parametric is True:
        # Calculate mean and std
        StatsTable["Grp0_Mean"] = np.nanmean(x0, axis=0)
        Mean095CI = (1.96 * (np.nanstd(x0, ddof=1, axis=0) / np.sqrt(x0.notnull().sum()))).values
        StatsTable["Grp0_Mean-95CI"] = list(zip(np.round(StatsTable["Grp0_Mean"] - Mean095CI, 2), np.round(StatsTable["Grp0_Mean"] + Mean095CI, 2)))
        StatsTable["Grp1_Mean"] = np.nanmean(x1, axis=0)
        Mean195CI = (1.96 * (np.nanstd(x1, ddof=1, axis=0) / np.sqrt(x1.notnull().sum()))).values
        StatsTable["Grp1_Mean-95CI"] = list(zip(np.round(StatsTable["Grp1_Mean"] - Mean195CI, 2), np.round(StatsTable["Grp1_Mean"] + Mean195CI, 2)))

        # Sign
        sign = np.nanmedian(x1, axis=0) / np.nanmedian(x0, axis=0)
        sign = np.where(sign > 1, 1, 0)
        StatsTable["Sign"] = sign

        # T test
        xnp = x.values
        t = scipy.stats.ttest_ind(x0, x1, nan_policy="omit")
        StatsTable["TTestStat"] = t[0]
        StatsTable["TTestPvalue"] = t[1]

        # BH correction
        bonTTest = multipletests(t[1], method="fdr_bh")
        # Round BH correction p-value
        StatsTable["bhQvalue"] = bonTTest[1]
    else:
        # Calculate median
        StatsTable["Grp0_Median"] = np.nanmedian(x0, axis=0)
        np.random.seed(seed)  # set seed for bootstrap
        x0_boot = []
        for i in range(100):
            x0_copy = x0.copy()
            x0_copy = x0_copy.iloc[np.random.randint(0, len(x0_copy), size=len(x0_copy))]
            x0_boot.append(np.nanmedian(x0_copy, axis=0))
        x0low = np.round(np.percentile(x0_boot, 2.5, axis=0), 2)
        x0high = np.round(np.percentile(x0_boot, 97.5, axis=0), 2)
        StatsTable["Grp0_Median-95CI"] = list(zip(x0low, x0high))

        StatsTable["Grp1_Median"] = np.nanmedian(x1, axis=0)
        np.random.seed(seed)  # set seed for bootstrap
        x1_boot = []
        for i in range(100):
            x1_copy = x1.copy()
            x1_copy = x1_copy.iloc[np.random.randint(0, len(x1_copy), size=len(x1_copy))]
            x1_boot.append(np.nanmedian(x1_copy, axis=0))
        x1low = np.round(np.percentile(x1_boot, 2.5, axis=0), 2)
        x1high = np.round(np.percentile(x1_boot, 97.5, axis=0), 2)
        StatsTable["Grp1_Median-95CI"] = list(zip(x1low, x1high))
        StatsTable["MedianFC"] = np.nanmedian(x1, axis=0) / np.nanmedian(x0, axis=0)

        # Sign
        sign = np.nanmedian(x1, axis=0) / np.nanmedian(x0, axis=0)
        sign = np.where(sign > 1, 1, 0)
        StatsTable["Sign"] = sign

        # Man-Whitney U
        m = []
        for i in range(len(x.T)):
            manwstat = scipy.stats.mannwhitneyu(x0.values[:, i], x1.values[:, i])
            m.append(manwstat)
        StatsTable["MannWhitneyU"] = [i[0] for i in m]
        StatsTable["MannWhitneyPvalue"] = [i[1] for i in m]

        # BH correction
        bonMannWhitney = multipletests(StatsTable["MannWhitneyPvalue"], method="fdr_bh")
        StatsTable["bhQvalue"] = bonMannWhitney[1]

    # Calculate total missing and total missing %
    nannum = x.isnull().sum()
    nanperc = nannum / x.shape[0]
    StatsTable["TotalMissing"] = nannum.values
    StatsTable["PercTotalMissing"] = np.round(nanperc.values * 100, 3)

    # Calculating missing % for group 0, and group 1...
    nanperc_0 = x0.isnull().sum() / len(x0)
    nanperc_1 = x1.isnull().sum() / len(x1)
    StatsTable["Grp0_Missing"] = np.round(nanperc_0.values * 100, 3)
    StatsTable["Grp1_Missing"] = np.round(nanperc_1.values * 100, 3)

    # Shapiro-Wilk
    s = []
    for i in range(len(x.T)):
        newx = x.values[:, i][~np.isnan(x.values[:, i])]
        shapstat = scipy.stats.shapiro(newx)
        s.append(shapstat)
    StatsTable["ShapiroW"] = [i[0] for i in s]  # Shapstat is a list of tuples, this is an approach to get this out of the tuple
    StatsTable["ShapiroPvalue"] = [i[1] for i in s]

    # Levenes
    l = []
    for i in range(len(x.T)):
        newx0 = x0.values[:, i][~np.isnan(x0.values[:, i])]
        newx1 = x1.values[:, i][~np.isnan(x1.values[:, i])]
        levstat = scipy.stats.levene(newx0, newx1)
        l.append(levstat)
    StatsTable["LeveneW"] = [i[0] for i in l]
    StatsTable["LevenePvalue"] = [i[1] for i in l]

    # Make the Idx column start from 1
    StatsTable.index = np.arange(1, len(StatsTable) + 1)
    return StatsTable
