import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from nearest_neighbor_matcher import NearestNeighborMatcher


def example_basic_matching():
    print("=" * 60)
    print("示例 1: 基本最近邻匹配")
    print("=" * 60)

    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )

    matcher = NearestNeighborMatcher(
        n_neighbors=1,
        metric="euclidean",
        with_replacement=False,
        standardize=True,
    )

    treated_idx, control_idx, distances = matcher.fit_transform(X, y)

    print(f"处理组样本数: {len(treated_idx)}")
    print(f"对照组样本数: {len(matcher.control_indices_)}")
    print(f"匹配成功对数: {len(treated_idx)}")
    print(f"平均匹配距离: {np.mean(distances):.4f}")
    print(f"最小匹配距离: {np.min(distances):.4f}")
    print(f"最大匹配距离: {np.max(distances):.4f}")
    print()


def example_k_neighbors():
    print("=" * 60)
    print("示例 2: K 近邻匹配 (k=3)")
    print("=" * 60)

    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_informative=3,
        weights=[0.8, 0.2],
        random_state=42,
    )

    print(f"处理组样本数: {np.sum(y == 1)}")
    print(f"对照组样本数: {np.sum(y == 0)}")

    matcher = NearestNeighborMatcher(
        n_neighbors=3,
        metric="euclidean",
        with_replacement=False,
        standardize=True,
    )

    treated_idx, control_idx, distances = matcher.fit_transform(X, y)

    print(f"匹配成功对数: {len(treated_idx)}")
    print(f"每个处理组匹配的对照数: {control_idx.shape[1]}")
    print(f"总共匹配的对照样本数: {control_idx.size}")
    print(f"平均匹配距离: {np.mean(distances):.4f}")
    print()


def example_with_replacement():
    print("=" * 60)
    print("示例 3: 有放回匹配")
    print("=" * 60)

    X, y = make_classification(
        n_samples=200,
        n_features=5,
        weights=[0.2, 0.8],
        random_state=42,
    )

    print(f"处理组样本数: {np.sum(y == 1)}")
    print(f"对照组样本数: {np.sum(y == 0)}")

    matcher = NearestNeighborMatcher(
        n_neighbors=1,
        metric="euclidean",
        with_replacement=True,
        standardize=True,
    )

    treated_idx, control_idx, distances = matcher.fit_transform(X, y)

    unique_controls = len(np.unique(control_idx))
    print(f"匹配成功对数: {len(treated_idx)}")
    print(f"使用的唯一对照样本数: {unique_controls}")
    print(f"平均匹配距离: {np.mean(distances):.4f}")
    print()


def example_dataframe_output():
    print("=" * 60)
    print("示例 4: DataFrame 格式输出")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame({
        "age": np.random.randint(20, 70, n_samples),
        "income": np.random.randint(30000, 150000, n_samples),
        "score": np.random.normal(50, 15, n_samples),
        "treated": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    })

    feature_cols = ["age", "income", "score"]
    X = df[feature_cols].values
    treated = df["treated"].values

    matcher = NearestNeighborMatcher(
        n_neighbors=1,
        metric="euclidean",
        with_replacement=False,
        standardize=True,
    )

    matcher.fit(X, treated)
    matcher.transform()

    matched_df = matcher.get_matched_dataframe(df, treated_col="treated")

    print("匹配结果前 10 行:")
    print(matched_df.head(10))
    print()
    print(f"匹配后总样本数: {len(matched_df)}")
    print(f"处理组样本数: {(matched_df['group'] == 'treated').sum()}")
    print(f"对照组样本数: {(matched_df['group'] == 'control').sum()}")
    print()


def example_propensity_score_matching():
    print("=" * 60)
    print("示例 5: 倾向性得分匹配 (简化版)")
    print("=" * 60)

    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)
    n_samples = 200

    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        random_state=42,
    )

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X, y)
    propensity_scores = log_reg.predict_proba(X)[:, 1]

    matcher = NearestNeighborMatcher(
        n_neighbors=1,
        metric="euclidean",
        with_replacement=False,
        standardize=False,
    )

    treated_idx, control_idx, distances = matcher.get_propensity_matched(
        propensity_scores=propensity_scores,
        treated=y,
        caliper=None,
    )

    print(f"处理组样本数: {len(treated_idx)}")
    print(f"匹配成功对数: {len(treated_idx)}")
    print(f"平均倾向性得分差: {np.mean(distances):.4f}")
    print(f"最小倾向性得分差: {np.min(distances):.4f}")
    print(f"最大倾向性得分差: {np.max(distances):.4f}")
    print()


def example_different_metrics():
    print("=" * 60)
    print("示例 6: 不同距离度量对比")
    print("=" * 60)

    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        random_state=42,
    )

    metrics = ["euclidean", "manhattan", "cosine"]

    for metric in metrics:
        matcher = NearestNeighborMatcher(
            n_neighbors=1,
            metric=metric,
            with_replacement=False,
            standardize=True,
        )

        _, _, distances = matcher.fit_transform(X, y)
        print(f"{metric:12s} - 平均距离: {np.mean(distances):.4f}, 最大距离: {np.max(distances):.4f}")

    print()


if __name__ == "__main__":
    example_basic_matching()
    example_k_neighbors()
    example_with_replacement()
    example_dataframe_output()
    example_propensity_score_matching()
    example_different_metrics()

    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
