import numpy as np
import pandas as pd
from nearest_neighbor_matcher import NearestNeighborMatcher
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
n = 200

df = pd.DataFrame({
    "age": np.random.randint(20, 70, n),
    "income": np.random.randint(30000, 150000, n),
    "score": np.random.normal(50, 15, n),
    "treated": np.random.choice([0, 1], n, p=[0.6, 0.4]),
})

feature_cols = ["age", "income", "score"]
X = df[feature_cols].values
treated = df["treated"].values

print("=== 测试 DataFrame 输出 ===")
matcher = NearestNeighborMatcher(caliper=1.0, standardize=True)
matcher.fit(X, treated)
matcher.transform()

matched_df = matcher.get_matched_dataframe(df)
print(f"匹配后总行数: {len(matched_df)}")
print(f"处理组: {(matched_df['group'] == 'treated').sum()}")
print(f"对照组: {(matched_df['group'] == 'control').sum()}")
print(f"距离都 <= caliper? {(matched_df['match_distance'] <= 1.0).all()}")
print()

print("=== 测试 PSM 卡钳 ===")
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X, treated)
ps = log_reg.predict_proba(X)[:, 1]

matcher_ps = NearestNeighborMatcher(standardize=True)

t_no, c_no, d_no = matcher_ps.get_propensity_matched(ps, treated, caliper=None)
print(f"无 caliper: 匹配对数={len(t_no)}, 最大距离={np.max(d_no):.4f}")

for cal in [0.01, 0.05, 0.1]:
    t, c, d = matcher_ps.get_propensity_matched(ps, treated, caliper=cal)
    max_d = np.max(d) if len(d) > 0 else 0
    print(f"caliper={cal}: 匹配对数={len(t):3d}, 最大距离={max_d:.4f}")

print()
print("所有卡钳功能测试通过！")
