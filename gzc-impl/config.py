from dataclasses import dataclass
import numpy as np

@dataclass
class ExperimentConfig:
    data_set: str #数据集名字 
    simulation_times: list #控制评估的精度（模拟次数）
    methods: list = None #选种子集的方法
    seed_num_list: list = None #种子集列表 # deprecated
    monte_carlo_L: int = 100 # 蒙特卡洛模拟次数
    distribution_type: str = 'random' # poisson gamma powerlaw random
    constant_factor_distri: str = 'random' # deprecated
    personalization: str = 'None'
    # data_prefix: str = '/home/wen/work/coup_spread'
    data_prefix: str = '/root/work/coupon/coup_spread'
    method_type: str = 'None'

    num_steps: int = 1 # deprecated
    scale_factor: int = 1000 # deprecated
    num_samples: int = 50000

    seeds_num: int = 1
    randomness_factor: float = 0.2

    rng: np.random.Generator = np.random.default_rng(1)

    single_sim_func: str = 'AgainReJudge'

    version: str = 'v0'

    single_file_switch: bool = False

    # === 新增：log_continuous 分布的控制参数 ===
    log_alpha_base: float = 0.01
    log_alpha_slope: float = 0.1   # 控制低度数节点的采纳率上限
    log_beta_base: float = 0.01
    log_beta_slope: float = 0.6   # 控制高度数节点的丢弃率上限

    @property
    def adj_file(self):
        return f"{self.data_prefix}/dataset/network/{self.data_set}-adj.pkl"

    @property
    def param_str(self):
        """生成参数后缀，用于区分不同参数下的缓存文件"""
        if self.distribution_type == 'log_continuous':
            return f"_aSlope{self.log_alpha_slope}_bSlope{self.log_beta_slope}"
        return ""

    # 当种子数和分布参数相同时 采用相同的概率分布
    def distribution_file(self, m = 0):
        # 加入 param_str 防止不同参数覆盖同一个分布文件
        return f"{self.data_prefix}/{self.data_set}/{self.version}/distribution-in-{self.data_set}/distribution-{self.distribution_type}{self.param_str}_seedNum-{m}.pkl"

    def deliverers_cache_file(self, method, m = 0):
        # 加入 param_str 防止读取到旧参数生成的种子
        return f"{self.data_prefix}/{self.data_set}/{self.version}/seeds-with-{self.data_set}/{self.distribution_type}{self.param_str}_{method}_seedNum-{m}_SSRNum-{self.num_samples}.txt"

    def usage_rate_file(self, m = 0):
        times = ",".join(str(time) for time in self.simulation_times)
        # 结果文件也加上 param_str，或者统一写到一个带 search 字样的文件里方便画图
        return f"{self.data_prefix}/gzc-impl/results/{self.data_set}/{self.version}/Search_{self.distribution_type}_SSRNum-{self.num_samples}_seedNum-{self.seeds_num}.csv"

    def log_file(self):
        times = ",".join(str(time) for time in self.simulation_times)
        if self.distribution_type == "powerlaw" or self.distribution_type == "gamma" or self.distribution_type == "poisson":
            return f"{self.data_prefix}/gzc-impl/logs/{self.data_set}/{self.version}/{self.distribution_type}_SSRNum-{self.num_samples}-_seedNum-{self.seeds_num}_simuTimes-{times}_t-s-d-baseValue-{self.tran_base_value, self.succ_base_value, self.dis_base_value}_t-s-d-factor-{self.tran_degree_influence_factor, self.succ_degree_influence_factor, self.dis_degree_influence_factor}.log"
        return f"{self.data_prefix}/gzc-impl/logs/{self.data_set}/{self.version}/{self.distribution_type}_SSRNum-{self.num_samples}-_seedNum-{self.seeds_num}_simuTimes-{times}.log"