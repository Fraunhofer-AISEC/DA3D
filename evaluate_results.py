import argparse
import math
import pandas as pd

from pathlib import Path
from typing import List, Tuple
from scipy.stats import wilcoxon, friedmanchisquare

from libs.constants import BASE_PATH

# Determine the mean and standard deviation of all result files - output is a LaTeX table
# python3 evaluate_results.py 110 210 310 410 510 610 710
# python3 evaluate_results.py 110 210 310 410 510 610 710 --is_ablation 1 --metric_name AUC --p_pollution 0.0 --out_path ./results/ablation.csv

# Configuration
this_parse = argparse.ArgumentParser(description="Merge the test result of all random seeds")
this_parse.add_argument(
    'random_seeds', nargs='+', help='Random seeds of the experiments'
)
this_parse.add_argument(
    '--p_pollution', nargs='+', help='Pollution factors to evaluate, if not set use 0.0 and 0.01'
)
this_parse.add_argument(
    '--n_anomalies', default=0, type=int, help='Number of known anomalies during training'
)
this_parse.add_argument(
    "--model_path", default=BASE_PATH / "models", type=Path, help="Path to the results (usually where the models are)"
)
this_parse.add_argument(
    "--metric_name", default=None, type=str, help="Name of the metric, usually 'AUC' or 'AP', if None show both"
)
this_parse.add_argument(
    "--p_name", default="wilcoxon", type=str, help="Significance test name, either 'wilcoxon' or 'friedman'"
)
this_parse.add_argument(
    "--show_stddev", default=True, type=bool, help="Show the standard deviation next to the mean"
)
this_parse.add_argument(
    "--is_ablation", default=False, type=bool, help="Results for the ablation study on different stddev"
)
this_parse.add_argument(
    "--show_pollution", default=False, type=bool, help="Show the pollution as separate column/row"
)
this_parse.add_argument(
    "--no_transpose", default=False, type=bool, help="If True, the data sets are on the rows and baselines on the columns"
)
this_parse.add_argument(
    "--out_path", default=None, type=Path, help="Path to output csv - if None, use stdout instead"
)
this_args = this_parse.parse_args()

AD_NAME = "DA3D"
P_POLLUTION = [0.0, 0.01] if not this_args.p_pollution else this_args.p_pollution
# In our ablation study, we compare different sample stddevs
if this_args.is_ablation:
    BASELINE_METHODS = ["DA3D-1", AD_NAME, "DA3D-3", "DA3D-4"]
else:
    BASELINE_METHODS = [AD_NAME, "TrivialSimpleDA3D", "DeepSVDD", "DAGMM", "REPEN", "fAnoGAN", "GANomaly", "FenceGAN", "A3"]

NAME_TO_ID = {
    # Unsupervised
    "Mammography_{SEED}_y_normal:0,y_anomalous:1,p_pollution:{POLL},n_train_anomalies:0": "Mammo.",
    "URL_{SEED}_y_normal:0,1,y_anomalous:Defacement,malware,phishing,spam,p_pollution:{POLL},n_train_anomalies:0": "URL",
    "KDD_{SEED}_y_normal:icmp,tcp,udp,y_anomalous:anomaly,p_pollution:{POLL},n_train_anomalies:0": "KDD",
    "CreditCard_{SEED}_y_normal:0,y_anomalous:1,p_pollution:{POLL},n_train_anomalies:0": "CC",
    "DoH_{SEED}_y_normal:2,3,50,51,y_anomalous:Malicious,p_pollution:{POLL},n_train_anomalies:0": "DoH",
    "Census_{SEED}_y_normal:Male,Female,y_anomalous:Anomalous,p_pollution:{POLL},n_train_anomalies:0": "Census",
    "CovType_{SEED}_y_normal:1,2,3,4,y_anomalous:5,6,7,p_pollution:{POLL},n_train_anomalies:0": "CoverT",
    "MNIST_{SEED}_y_normal:0,1,2,3,4,y_anomalous:5,6,7,8,9,p_pollution:{POLL},n_train_anomalies:0": "MNIST",
    "FMNIST_{SEED}_y_normal:0,1,2,3,4,y_anomalous:5,6,7,8,9,p_pollution:{POLL},n_train_anomalies:0": "FMNIST",
    "EMNIST_{SEED}_y_normal:1,2,3,4,5,6,7,8,9,10,11,12,13,y_anomalous:14,15,16,17,18,19,20,21,22,23,24,25,26,p_pollution:{POLL},n_train_anomalies:0": "EMNIST",
}


def get_path(basepath: Path, p_pollution: float, random_seed: int, n_anomalies: int, file_name: str, file_suffix: str = ".metric.csv"):
    out_path = basepath
    # There are subfolders based on the pollution and the random seed
    out_path /= f"{p_pollution}_{random_seed}"
    # The filename contains the random seed (bad design decision btw)
    parsed_name = file_name.replace("{SEED}", str(random_seed))
    # And also the pollution (honestly, this makes things harder)
    parsed_name = parsed_name.replace("{POLL}", str(p_pollution).replace(".", ""))
    parsed_name = parsed_name.replace("{NANO}", str(n_anomalies))
    out_path /= parsed_name
    out_path = out_path.with_suffix(file_suffix)

    return out_path


if __name__ == '__main__':
    # In the end, we want a DF with all results indexed by the contamination and the experiment IDs
    df_tot = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [P_POLLUTION, list(NAME_TO_ID.values())],
            names=["Cont.", "Exp."]
        )
    )
    # We need the single experiments for the friedman coefficient
    df_out_per_rep = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [P_POLLUTION, list(NAME_TO_ID.values()), ["AUC", "AP"]],
            names=["Cont.", "Exp.", "Metric"]
        ),
        index=pd.MultiIndex.from_product(
            [BASELINE_METHODS, this_args.random_seeds],
            names=["Baseline", "Seed"]
        )
    )
    df_out = df_tot.copy()
    df_out.index = pd.MultiIndex(levels=2*[[]], codes=2*[[]], names=["Method", "Metric"])

    # Go through all metric files
    results_by_method = {}
    for cur_pollution in P_POLLUTION:
        for cur_name, cur_id in NAME_TO_ID.items():

            # We open all metric files given their random seed
            all_metrics = []
            for cur_seed in this_args.random_seeds:
                cur_path = get_path(
                    basepath=this_args.model_path,
                    p_pollution=cur_pollution,
                    n_anomalies=this_args.n_anomalies,
                    random_seed=cur_seed,
                    file_name=cur_name
                )
                # List all files that are missing
                try:
                    in_df = pd.read_csv(cur_path, index_col=0)
                    all_metrics.append(in_df)
                    df_out_per_rep.loc[(BASELINE_METHODS, cur_seed), (cur_pollution, NAME_TO_ID[cur_name], ["AUC", "AP"])] = in_df.loc[BASELINE_METHODS, :] .values
                except FileNotFoundError:
                    print(f"Cannot find {cur_path}. Please check the path.")
                    continue

            # Once opened, we merge them
            pd_concat = pd.concat(all_metrics)
            concat_by_method = pd_concat.groupby(pd_concat.index)
            # We want everything in one series which will become a row in the final DF
            this_mean = concat_by_method.mean().stack()
            this_std = concat_by_method.std().stack()
            # Also we should add a new level to the MultiIndex to mark the mean and stddev
            this_mean = pd.DataFrame(this_mean)
            this_mean.loc[:, "type"] = "mean"
            this_mean = this_mean.set_index("type", append=True)
            this_std = pd.DataFrame(this_std)
            this_std.loc[:, "type"] = "std"
            this_std = this_std.set_index("type", append=True)

            # Add to the overall DF
            merged_metric = pd.concat([this_mean, this_std])
            df_tot[(cur_pollution, NAME_TO_ID[cur_name])] = merged_metric[0]

    all_baselines = df_tot.index.unique(0)
    # If not given otherwise, the AUC is our main metric
    if this_args.metric_name is None:
        df_tot = df_tot.reindex(["AUC", "AP"], axis=0, level=1)
    else:
        df_tot = df_tot.loc[(all_baselines, [this_args.metric_name]), :]
    # Reorder baselines
    level_name_reordered = BASELINE_METHODS
    # Reorder
    df_tot = df_tot.reindex(level_name_reordered, axis=0, level=0)

    # Round
    df_not_rounded = df_tot.copy()
    df_tot = df_tot.round(decimals=2)
    # Save if desired
    if this_args.out_path is not None:
        df_csv = df_not_rounded.copy()
        # Combine the MultiIndex for easier indexing afterwards
        df_csv.index = df_csv.index.to_series().apply(lambda x: f"{x[0]}-{x[1]}-{x[2]}")
        df_csv = df_csv.transpose()
        # Save
        df_csv.to_csv(this_args.out_path)

    # Decision: let's build the LaTeX code here instead of using pgfplotstable & Co
    for cur_idx, cur_df in df_tot.groupby(level=[0, 1]):
        # Merge to "mean \pm stddev"
        this_latex = cur_df.iloc[0, :].map("{:,.2f}".format)
        if this_args.show_stddev:
            this_latex += "\\scriptscriptstyle \\pm " + cur_df.iloc[1, :].map("{:,.2f}".format)
        # Add the math environment
        this_latex = "$" + this_latex + "$"

        # Highest score should be black, rest gray
        max_per_column = df_tot.loc[(slice(None), cur_idx[1], "mean"), :].max(axis=0)
        is_max = cur_df.loc[cur_idx + ("mean",), :] == max_per_column
        this_latex.loc[is_max] = "\\color{black}" + this_latex.loc[is_max]
        # this_latex.loc[is_max] = "\\textbf{" + this_latex.loc[is_max] + "}"

        df_out.loc[cur_idx, :] = this_latex

    # Add p-value
    if this_args.p_name == "wilcoxon":
        for cur_idx, cur_df in df_not_rounded.groupby(axis=0, level=[0, 1]):
            # Don't compare GAA to itself
            if cur_idx[0] == AD_NAME:
                continue

            # First group by baseline & metric, then by contamination level
            for cur_idx_2, cur_df_2 in cur_df.groupby(axis=1, level=0):
                # Compare the distribution of GAA to the baseline
                dist_GAA = df_not_rounded.loc[(AD_NAME, cur_idx[1], "mean"), (cur_idx_2, slice(None))]
                dist_baseline = cur_df_2.loc[cur_idx + ("mean", ), :]
                # Calculate the p-value
                _, p_val = wilcoxon(x=dist_GAA, y=dist_baseline)

                # Prepare for LaTeX
                p_val = round(p_val, ndigits=2)
                p_val = f"${p_val:,.2f}$"
                # Add to the output DF
                df_out.loc[cur_idx, (cur_idx_2, "p-val")] = p_val
                pass
        # Mark GAA's p-value by "-"
        df_out.loc[(AD_NAME, slice(None)), (slice(None), "p-val")] = "-"
    elif this_args.p_name == "friedman":
        # We need the results of one method for all datasets
        # 0) Loop over pollution and metric
        for cur_id_0, cur_df_0 in df_out_per_rep.groupby(axis=1, level=[0, 2]):
            # 1) Loop over method
            for cur_id_1, cur_df_1 in cur_df_0.groupby(axis=0, level=0):
                list_of_results = []
                # 2) Loop over seed
                for cur_id_2, cur_df_2 in cur_df_1.groupby(axis=0, level=1):
                    list_of_results.append(cur_df_2.values.reshape((-1,)))
                # Calculate the Friedman score
                friedman_score = friedmanchisquare(*list_of_results)
                # Prepare for LaTeX
                p_val = round(friedman_score.pvalue, ndigits=2)
                p_val = f"${p_val:,.2f}$"
                # Add to the output df
                df_out.loc[(cur_id_1, cur_id_0[1]), (cur_id_0[0], "p-val")] = p_val
    elif this_args.p_name is None:
        # Don't add anything
        pass
    else:
        raise NotImplementedError("Unknown significance test.")

    # Add average row
    for cur_idx, cur_df in df_not_rounded.groupby(axis=1, level=0):
        df_avg = cur_df.mean(axis=1).round(decimals=2)
        # We're only interested in the mean of the mean
        df_avg = df_avg.loc[(slice(None), slice(None), "mean")]
        # Add the math environment
        this_latex = "$" + df_avg.map("{:,.2f}".format) + "$"
        # Show the maximum
        all_max = []
        for max_idx, max_df in df_avg.groupby(axis=0, level=1):
            all_max.append(
                max_df.loc[(slice(None), [max_idx])] == max_df.max()
            )
        is_max = pd.concat(all_max)
        this_latex.loc[is_max] = "\\color{black}" + this_latex.loc[is_max]
        df_out[(cur_idx, "mean")] = this_latex

    # Convert to TeX
    df_out = df_out.sort_index(axis=1, level=0)
    if not this_args.no_transpose:
        df_out = df_out.transpose()
    latex = df_out.to_latex(
        multicolumn_format="c", column_format=">{\\color{gray}}c "*(df_out.index.nlevels + len(df_out.columns)), escape=False
    )
    # Get back the backslash and math environments
    latex = latex.replace("\\textbackslash ", "\\").replace("\\$", "$").replace("0.", ".")

    if this_args.out_path is None:
        print(latex)
