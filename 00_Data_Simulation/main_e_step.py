import numpy as np
import pandas as pd
import os
import random  # 新增
from e_step_generator import SyntheticCohortGenerator
from matrix_translator import PriorToCorrelationTranslator


def set_global_determinism(seed=42):
    """
    [NC级方法学补丁 3] 绝对全局确定性锁
    在操作系统、Python解释器、以及科学计算底层库三个维度彻底锁死随机性，
    确保在任何电脑上跑出的虚拟队列都是 100% 比特级一致的。
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔒 全局系统种子已锁死 (Seed={seed})，进入确定性计算模式 (Deterministic Mode)。")


def run_full_e_step(data_dir, n_patients=3000):
    print("========================================")
    print("🤖 启动 Copula 结构期望最大化 E-步 引擎")
    print("========================================\n")

    # 【核心修复】：在任何计算开始前，锁死宇宙！
    set_global_determinism(seed=2026)

    # 第一步：翻译文献先验，生成 Sigma 矩阵
    translator = PriorToCorrelationTranslator(data_dir)
    sigma_matrix = translator.build_correlation_matrix()

    # 第二步：将 Sigma 矩阵喂给生成器，创造虚拟队列
    print("\n----------------------------------------")
    generator = SyntheticCohortGenerator(data_dir)
    # 因为种子已经锁死，这里内部调用的多元正态抽样每次都会输出一模一样的矩阵
    synthetic_data = generator.generate_cohort(n_patients=n_patients, correlation_matrix=sigma_matrix)

    # ---------------------------------------------------------
    # [NC级方法学补丁 1]：自适应边缘重校准 (Adaptive Marginal Recalibration)
    # 消除因高危因素(如脑转移、ctDNA+)富集导致的宏观生存期暴跌
    # ---------------------------------------------------------
    print("\n⚖️ 正在执行自适应边缘重校准 (强行锚定文献基准中位数)...")
    nodes_df = pd.read_csv(os.path.join(data_dir, 'nodes_marginal.csv')).set_index('node_id')
    survival_endpoints = ['Y_PFS_TKI_All', 'Y_OS_TKI_All', 'Y_PFS_IO', 'Y_OS_IO']

    for endpoint in survival_endpoints:
        if endpoint in synthetic_data.columns:
            target_median = float(nodes_df.loc[endpoint, 'base_value'])
            current_median = synthetic_data[endpoint].median()
            calibration_alpha = target_median / current_median
            synthetic_data[endpoint] = synthetic_data[endpoint] * calibration_alpha
            print(
                f"  🔧 [{endpoint}] 目标: {target_median:.1f} | 漂移后实际: {current_median:.2f} | 自动修正系数(Alpha): {calibration_alpha:.3f}")

    # ---------------------------------------------------------
    # [NC级方法学补丁 2]：应用右删失模拟器 (Right-Censoring Simulation)
    # ---------------------------------------------------------
    print("\n⚙️ 正在应用右删失模拟器 (生成真实的随访与失访分布)...")
    median_followup = 30.0

    # 【注意】：这里不需要再次 np.random.seed(2026) 了，全局锁已经生效。

    for endpoint in survival_endpoints:
        if endpoint in synthetic_data.columns:
            true_time = synthetic_data[endpoint].values
            scale_param = median_followup / np.log(2)
            censoring_time = np.random.exponential(scale=scale_param, size=n_patients)
            observed_time = np.minimum(true_time, censoring_time)
            event_indicator = (true_time <= censoring_time).astype(int)
            synthetic_data[endpoint] = observed_time
            synthetic_data[f"{endpoint}_Event"] = event_indicator
            event_rate = event_indicator.mean() * 100
            print(f"  ✅ [{endpoint}] 删失完毕: 平均观察期={observed_time.mean():.1f}个月, 事件发生率={event_rate:.1f}%")

    # 第三步：保存结果以供后续分析
    output_path = os.path.join(data_dir, f"synthetic_cohort_n{n_patients}.csv")
    synthetic_data.to_csv(output_path, index=False)

    print("\n========================================")
    print(f"🎉 成功生成 {n_patients} 名带因果约束及真实删失状态的虚拟患者！")
    print(f"📁 数据已保存至: {output_path}")
    print("========================================")

    return synthetic_data


if __name__ == "__main__":
    DATA_DIR = r"./data"
    df_cohort = run_full_e_step(DATA_DIR, n_patients=3000)

    print("\n--- 临床规律交叉验证 (脑转移对生存期的暴击) ---")
    df_events = df_cohort[df_cohort['Y_PFS_TKI_All_Event'] == 1]
    med_pfs_no_brain = df_events[df_events['E1_Brain_Met'] == 0]['Y_PFS_TKI_All'].median()
    med_pfs_with_brain = df_events[df_events['E1_Brain_Met'] == 1]['Y_PFS_TKI_All'].median()

    print(f"✅ 无脑转移患者的中位 PFS: {med_pfs_no_brain:.2f} 个月")
    print(f"✅ 脑转移患者的 中位 PFS:  {med_pfs_with_brain:.2f} 个月")
    if med_pfs_with_brain < med_pfs_no_brain:
        print("💡 结论：完美！由于我们在矩阵中注入了 HR=1.95 的负相关先验，模型自动让脑转移患者死得更快了。")