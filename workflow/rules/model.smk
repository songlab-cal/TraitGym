rule run_classifier:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        lambda wildcards: expand("results/dataset/{{dataset}}/features/{features}.parquet", features=config["feature_sets"][wildcards.feature_set]),
    output:
        "results/dataset/{dataset}/preds/{subset}/{feature_set}.{classifier,LogisticRegression|RidgeRegression|RandomForest|XGBoost|PCALogisticRegression|FeatureSelectionLogisticRegression|BestFeature}.{split_mode,chrom|odd_even|pos_quarter}.parquet",
    threads:
        lambda wildcards: 1 if wildcards.classifier == "BestFeature" else workflow.cores
    run:
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        all_features = []
        for features, path in zip(config["feature_sets"][wildcards.feature_set], input[2:]):
            df = pd.read_parquet(path)
            df.columns = [f"{features}_{col}" for col in df.columns]
            all_features += df.columns.tolist()
            V = pd.concat([V, df], axis=1)
        V = subset.merge(V, on=COORDINATES, how="left")
        print(V)

        mask_train_list = []
        
        if wildcards.split_mode == "chrom":
            for chrom in V.chrom.unique():
                mask_train = V.chrom != chrom
                mask_train_list.append(mask_train)
        elif wildcards.split_mode == "odd_even":
            for chroms in ODD_EVEN_CHROMS:
                mask_train = V.chrom.isin(chroms)
                mask_train_list.append(mask_train)
        elif wildcards.split_mode == "pos_quarter":
            pos_subsets = np.array_split(np.unique(V.pos), 4)
            for pos_subset in pos_subsets:
                mask_train = V.pos.isin(pos_subset)
                mask_train_list.append(mask_train)

        for mask_train in tqdm(mask_train_list):
            mask_test = ~mask_train
            V.loc[mask_test, "score"] = train_predict(
                V[mask_train], V[mask_test], all_features,
                classifier_map[wildcards.classifier]
            )

        V[["score"]].to_parquet(output[0], index=False)


rule subset_classifier:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/preds/all/{model}.parquet",
    output:
        "results/dataset/{dataset}/preds/{subset}/{model}.subset_from_all.parquet",
    run:
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        df = pd.read_parquet(input[2])
        V = pd.concat([V, df], axis=1)
        V = subset.merge(V, on=COORDINATES, how="left")
        V[["score"]].to_parquet(output[0], index=False)


rule unsupervised_pred:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/features/{features}.parquet",
    output:
        "results/dataset/{dataset}/preds/{subset}/{features}.{sign,plus|minus}.{feature}.parquet",
    run:
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        df = (
            pd.read_parquet(input[2], columns=[wildcards.feature])
            .rename(columns={wildcards.feature: "score"})
        )
        if wildcards.sign == "minus":
            df.score = -df.score
        V = pd.concat([V, df], axis=1)
        V = subset.merge(V, on=COORDINATES, how="left")
        assert V.score.isna().sum() == 0
        V[["score"]].to_parquet(output[0], index=False)


rule eval_unsupervised_features:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/features/{features}.parquet",
    output:
        "results/dataset/{dataset}/unsupervised_metrics/{subset}/{features}.csv",
    run:
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        df = pd.read_parquet(input[2])
        features = df.columns
        df = df.fillna(df.mean())
        V = pd.concat([V, df], axis=1)
        V = subset.merge(V, on=COORDINATES, how="left")
        balanced = V.label.sum() == len(V) // 2
        metric = roc_auc_score if balanced else average_precision_score
        metric_name = "AUROC" if balanced else "AUPRC"
        res = []
        for f in tqdm(features):
            score_plus = metric(V.label, V[f])
            score_minus = metric(V.label, -V[f])
            score = max(score_plus, score_minus)
            sign = "plus" if score_plus > score_minus else "minus"
            res.append([score, f, sign])
        res = pd.DataFrame(res, columns=[metric_name, "feature", "sign"])
        res = res.sort_values(metric_name, ascending=False)
        res.to_csv(output[0], index=False)


rule classifier_coefficients:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        lambda wildcards: expand("results/dataset/{{dataset}}/features/{features}.parquet", features=config["feature_sets"][wildcards.feature_set]),
    output:
        "results/dataset/{dataset}/coefficients/{subset}/{feature_set}.csv",
    threads:
        workflow.cores
    run:
        2 + 2
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        all_features = []
        for features, path in zip(config["feature_sets"][wildcards.feature_set], input[2:]):
            df = pd.read_parquet(path)
            df.columns = [f"{features}_{col}" for col in df.columns]
            all_features += df.columns.tolist()
            V = pd.concat([V, df], axis=1)
        V = subset.merge(V, on=COORDINATES, how="left")
        clf = train_lasso_logistic_regression(V[all_features], V.label, V.chrom)
        linear = clf.best_estimator_.named_steps["linear"]
        res = pd.DataFrame({
            "feature": all_features,
            "coef": linear.coef_[0],
        }).sort_values("coef", ascending=False, key=abs)
        res.to_csv(output[0], index=False)


rule get_metrics:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/preds/{subset}/{model}.parquet",
    output:
        "results/dataset/{dataset}/metrics/{subset}/{model}.csv",
    run:
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        V = subset.merge(V, on=COORDINATES, how="left")
        V["score"] = pd.read_parquet(input[2])["score"]
        balanced = V.label.sum() == len(V) // 2
        metric = roc_auc_score if balanced else average_precision_score
        metric_name = "AUROC" if balanced else "AUPRC"
        block = "chrom" if len(V.chrom.unique()) > 1 else "match_group"
        res = pd.DataFrame({
            "model": [wildcards.model],
            "metric": [metric_name],
            "score": [metric(V.label, V.score)],
            "se": [block_bootstrap_se(metric, V, "label", "score", block)],
        })
        res.to_csv(output[0], index=False)


#rule merge_metrics:
#    input:
#        lambda wildcards: expand(
#            "results/dataset/{{dataset}}/metrics/{model}.csv",
#            model=config["dataset_eval_models"][wildcards.dataset]
#        )
#    output:
#        "results/dataset/{dataset}/merged_metrics.csv",
#    run:
#        dfs = [pd.read_csv(f) for f in input]
#        df = pd.concat(dfs, axis=0)
#        df = df.sort_values(df.columns[-1], ascending=False)
#        print(df)
#        df.to_csv(output[0], index=False)
#
#
#rule plot_metrics:
#    input:
#        "results/dataset/{dataset}/test.parquet",
#        "results/dataset/{dataset}/merged_metrics.csv",
#    output:
#        "results/dataset/{dataset}/plot.pdf",
#    run:
#        V = pd.read_parquet(input[0])
#        n_pos, n_neg = V.label.sum(), len(V) - V.label.sum()
#        res = pd.read_csv(input[1])
#        metric = res.columns[-1]
#        if metric == "AUROC":
#            baseline = 0.5
#        elif metric == "AUPRC":
#            baseline = n_pos / (n_pos + n_neg)
#        plt.figure(figsize=(2,2))
#        g = sns.barplot(
#            data=res,
#            y="Model",
#            x=metric,
#            color="C0",
#        )
#        sns.despine()
#        sample_size = f"n={format_number(n_pos)} vs. {format_number(n_neg)}"
#        g.set(xlim=baseline, ylabel="")
#        title = f"{wildcards.dataset.split('/')[-1]}\n{sample_size}"
#        plt.title(title)
#        for bar, model in zip(g.patches, res.Model):
#            text = f'{bar.get_width():.3f}'
#
#            g.text(
#                max(bar.get_width(), baseline),  # X position, here at the end of the bar
#                bar.get_y() + bar.get_height()/2,  # Y position, in the middle of the bar
#                text,  # Text to be displayed, formatted to 3 decimal places
#                va='center'  # Vertical alignment
#            )
#        plt.savefig(output[0], bbox_inches="tight")


rule get_metrics_by_chrom:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/preds/{subset}/{model}.parquet",
    output:
        "results/dataset/{dataset}/metrics_by_chrom/{subset}/{model}.csv",
    run:
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        V = subset.merge(V, on=COORDINATES, how="left")
        V["score"] = pd.read_parquet(input[2])["score"]
        balanced = V.label.sum() == len(V) // 2
        metric = roc_auc_score if balanced else average_precision_score
        metric_name = "AUROC" if balanced else "AUPRC"
        res = []
        for chrom in V.chrom.unique():
            V_chrom = V[V.chrom == chrom]
            res.append([chrom, len(V_chrom), wildcards.model, metric(V_chrom.label, V_chrom.score)])
        res = pd.DataFrame(res, columns=["chrom", "n", "Model", metric_name])
        res.to_csv(output[0], index=False)


rule get_metrics_by_chrom_weighted_average:
    input:
        "results/dataset/{dataset}/metrics_by_chrom/{subset}/{model}.csv",
    output:
        "results/dataset/{dataset}/metrics_by_chrom_weighted_average/{subset}/{model}.csv",
    run:
        res = pl.read_csv(input[0])
        metric = res.columns[-1]

        def stat(df):
            x = df[metric]
            weight = df["n"] / df["n"].sum()
            return (x * weight).sum()

        res = pl.DataFrame({
            "model": [wildcards.model],
            "metric": [metric],
            "score": [stat(res)],
            "se": [bootstrap_se(res, stat)],
        })
        res.write_csv(output[0])


#rule merge_metrics_by_chrom:
#    input:
#        lambda wildcards: expand(
#            "results/dataset/{{dataset}}/metrics_by_chrom/{model}.csv",
#            model=config["dataset_eval_models"][wildcards.dataset]
#        )
#    output:
#        "results/dataset/{dataset}/merged_metrics_by_chrom.csv",
#    run:
#        dfs = [pd.read_csv(f) for f in input]
#        df = pd.concat(dfs, axis=0)
#        print(df)
#        df.to_csv(output[0], index=False)
#
#
#rule plot_metrics_by_chrom:
#    input:
#        "results/dataset/{dataset}/test.parquet",
#        "results/dataset/{dataset}/merged_metrics_by_chrom.csv",
#    output:
#        "results/dataset/{dataset}/plot_by_chrom.pdf",
#    run:
#        V = pd.read_parquet(input[0])
#        n_pos, n_neg = V.label.sum(), len(V) - V.label.sum()
#        res = pd.read_csv(input[1])
#        metric = res.columns[-1]
#        if metric == "AUROC":
#            baseline = 0.5
#        elif metric == "AUPRC":
#            baseline = n_pos / (n_pos + n_neg)
#        plt.figure(figsize=(2,2))
#        g = sns.boxplot(
#            data=res,
#            y="Model",
#            x=metric,
#            color="C0",
#            order=res.groupby("Model")[metric].median().sort_values(ascending=False).index,
#        )
#        sns.despine()
#        sample_size = f"n={format_number(n_pos)} vs. {format_number(n_neg)}"
#        g.set(
#            #xlim=baseline,
#            ylabel=""
#        )
#        title = f"{wildcards.dataset.split('/')[-1]}\n{sample_size}"
#        plt.title(title)
#        plt.savefig(output[0], bbox_inches="tight")
#


rule get_metrics_by_odd_even:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/preds/{subset}/{model}.parquet",
    output:
        "results/dataset/{dataset}/metrics_by_odd_even/{subset}/{model}.csv",
    run:
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        V = subset.merge(V, on=COORDINATES, how="left")
        V["score"] = pd.read_parquet(input[2])["score"]
        balanced = V.label.sum() == len(V) // 2
        metric = roc_auc_score if balanced else average_precision_score
        metric_name = "AUROC" if balanced else "AUPRC"
        res = []
        for name, chroms in zip(["odd", "even"], ODD_EVEN_CHROMS):
            V_chroms = V[V.chrom.isin(chroms)]
            res.append([name, len(V_chroms), wildcards.model, metric(V_chroms.label, V_chroms.score)])
        res = pd.DataFrame(res, columns=["chroms", "n", "Model", metric_name])
        res.to_csv(output[0], index=False)


rule get_metrics_by_odd_even_weighted_average:
    input:
        "results/dataset/{dataset}/metrics_by_odd_even/{subset}/{model}.csv",
    output:
        "results/dataset/{dataset}/metrics_by_odd_even_weighted_average/{subset}/{model}.csv",
    run:
        res = pd.read_csv(input[0])
        metric = res.columns[-1]
        res["weight"] = res.n / res.n.sum()
        res = pd.DataFrame({
            "Model": [wildcards.model],
            metric: [(res[metric] * res.weight).sum()]
        })
        res.to_csv(output[0], index=False)


rule get_metrics_by_pos_quarter:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/preds/{subset}/{model}.parquet",
    output:
        "results/dataset/{dataset}/metrics_by_pos_quarter/{subset}/{model}.csv",
    run:
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        V = subset.merge(V, on=COORDINATES, how="left")
        V["score"] = pd.read_parquet(input[2])["score"]
        metric = lambda y_true, y_pred: pearsonr(y_true, y_pred)[0]
        metric_name = "Pearson"
        pos_subsets = np.array_split(np.unique(V.pos), 4)
        res = []
        for name, pos_subset in zip(range(len(pos_subsets)), pos_subsets):
            V_pos = V[V.pos.isin(pos_subset)]
            res.append([name, len(V_pos), wildcards.model, metric(V_pos.label, V_pos.score)])
        res = pd.DataFrame(res, columns=["pos_quarter", "n", "Model", metric_name])
        res.to_csv(output[0], index=False)


rule get_metrics_by_pos_quarter_weighted_average:
    input:
        "results/dataset/{dataset}/metrics_by_pos_quarter/{subset}/{model}.csv",
    output:
        "results/dataset/{dataset}/metrics_by_pos_quarter_weighted_average/{subset}/{model}.csv",
    run:
        res = pd.read_csv(input[0])
        metric = res.columns[-1]
        res["weight"] = res.n / res.n.sum()
        res = pd.DataFrame({
            "Model": [wildcards.model],
            metric: [(res[metric] * res.weight).sum()]
        })
        res.to_csv(output[0], index=False)


rule get_mean_accuracy:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/subset/{subset}.parquet",
        "results/dataset/{dataset}/preds/{subset}/{model}.parquet",
    output:
        "results/dataset/{dataset}/mean_accuracy/{subset}/{model}.csv",
    run:
        V = pd.read_parquet(input[0])
        subset = pd.read_parquet(input[1])
        V = subset.merge(V, on=COORDINATES, how="left")
        V["score"] = pd.read_parquet(input[2])["score"]
        balanced = V.label.sum() == len(V) // 2
        assert balanced
        metric_name = "Mean accuracy"
        res = V.groupby("match_group").apply(lambda df: df.sort_values("score", ascending=False).label.iloc[0]).mean()
        print(res)
        raise Exception("debug")
        res = []
        for chrom in V.chrom.unique():
            V_chrom = V[V.chrom == chrom]
            res.append([chrom, len(V_chrom), wildcards.model, metric(V_chrom.label, V_chrom.score)])
        res = pd.DataFrame(res, columns=["chrom", "n", "Model", metric_name])
        res.to_csv(output[0], index=False)