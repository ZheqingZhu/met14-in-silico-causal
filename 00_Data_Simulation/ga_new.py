import numpy as np
import pandas as pd
import os
import random
from e_step_generator import SyntheticCohortGenerator
from matrix_translator import PriorToCorrelationTranslator


def set_global_determinism(seed):
    """
    [修改点 1]：让锁接受外部传入的动态种子
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔒 全局系统种子已锁死 (Seed={seed})，进入确定性计算模式。")


def run_full_e_step(data_dir, n_patients=3000, current_seed=2026):
    """
    [修改点 2]：主函数增加 current_seed 参数
    """
    print("\n" + "=" * 50)
    print(f"🤖 启动 Copula E-步 引擎 | 当前平行宇宙种子: {current_seed}")
    print("=" * 50 + "\n")

    # 【核心修复】：根据传入的种子，锁死当前宇宙！
    set_global_determinism(seed=current_seed)

    # 第一步：翻译文献先验，生成 Sigma 矩阵
    translator = PriorToCorrelationTranslator(data_dir)
    sigma_matrix = translator.build_correlation_matrix()

    # 第二步：创造虚拟队列
    print("\n----------------------------------------")
    generator = SyntheticCohortGenerator(data_dir)
    synthetic_data = generator.generate_cohort(n_patients=n_patients, correlation_matrix=sigma_matrix)

    # --- [NC级方法学补丁 1]：自适应边缘重校准 ---
    print("\n⚖️ 正在执行自适应边缘重校准...")
    nodes_df = pd.read_csv(os.path.join(data_dir, 'nodes_marginal.csv')).set_index('node_id')
    survival_endpoints = ['Y_PFS_TKI_All', 'Y_OS_TKI_All', 'Y_PFS_IO', 'Y_OS_IO']

    for endpoint in survival_endpoints:
        if endpoint in synthetic_data.columns:
            target_median = float(nodes_df.loc[endpoint, 'base_value'])
            current_median = synthetic_data[endpoint].median()
            calibration_alpha = target_median / current_median
            synthetic_data[endpoint] = synthetic_data[endpoint] * calibration_alpha

    # --- [NC级方法学补丁 2]：应用右删失模拟器 ---
    print("\n⚙️ 正在应用右删失模拟器...")
    median_followup = 30.0

    for endpoint in survival_endpoints:
        if endpoint in synthetic_data.columns:
            true_time = synthetic_data[endpoint].values
            scale_param = median_followup / np.log(2)
            censoring_time = np.random.exponential(scale=scale_param, size=n_patients)
            observed_time = np.minimum(true_time, censoring_time)
            event_indicator = (true_time <= censoring_time).astype(int)
            synthetic_data[endpoint] = observed_time
            synthetic_data[f"{endpoint}_Event"] = event_indicator

    # [修改点 3]：修改输出文件名，加上 seed，防止覆盖掉你的黄金数据！
    output_filename = f"synthetic_cohort_n{n_patients}_seed{current_seed}.csv"
    output_path = os.path.join(data_dir, output_filename)
    synthetic_data.to_csv(output_path, index=False)

    print("\n========================================")
    print(f"🎉 成功生成 3000 名虚拟患者！数据已保存至: {output_filename}")
    print("========================================")

    return synthetic_data


if __name__ == "__main__":
    DATA_DIR = r"./data"

    # ==========================================
    # 🧪 极限压力测试区 (Stress Testing)
    # 你可以在这里输入一个包含多个种子的列表来循环跑！
    # ==========================================
    test_seeds = [100]

    for seed in test_seeds:
        df_cohort = run_full_e_step(DATA_DIR, n_patients=3000, current_seed=seed)

        print(f"\n--- 验证宇宙 {seed} 的宏观规律 (脑转移) ---")
        df_events = df_cohort[df_cohort['Y_PFS_TKI_All_Event'] == 1]
        med_pfs_no_brain = df_events[df_events['E1_Brain_Met'] == 0]['Y_PFS_TKI_All'].median()
        med_pfs_with_brain = df_events[df_events['E1_Brain_Met'] == 1]['Y_PFS_TKI_All'].median()

        print(f"✅ [Seed {seed}] 无脑转移患者 中位PFS: {med_pfs_no_brain:.2f} 个月")
        print(f"✅ [Seed {seed}] 有脑转移患者 中位PFS: {med_pfs_with_brain:.2f} 个月\n")