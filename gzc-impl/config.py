from dataclasses import dataclass
import numpy as np

@dataclass
class ExperimentConfig:
    data_set: str #数据集名字 
    simulation_times: list #控制评估的精度（模拟次数）
    methods: list = None #选种子集的方法
    seed_num_list: list = None #种子集列表 # deprecated
    monte_carlo_L: int = 10 # 蒙特卡洛模拟次数
    distribution_type: str = 'random' # poisson gamma powerlaw random
    constant_factor_distri: str = 'random' # deprecated
    personalization: str = 'None'
    data_prefix: str = '/home/wen/pythonspace/data-test'
    method_type: str = 'None'

    num_steps: int = 1 # deprecated
    scale_factor: int = 1000 # deprecated
    num_samples: int = 50000

    seeds_num: int = 1
    tran_base_value: float = 1.0
    tran_degree_influence_factor: float = -4.5
    succ_base_value: float = 1.0
    succ_degree_influence_factor: float = 3.0
    dis_base_value: float = 1.0
    dis_degree_influence_factor: float = 2.0

    rng: np.random.Generator = np.random.default_rng(1)# 保证复现性




    
    @property
    def adj_file(self):
        return f"{self.data_prefix}/datasets/{self.data_set}-adj.pkl"

    def distribution_file(self, m = 0):
        return f"{self.data_prefix}/{self.data_set}/distribution-in-{self.data_set}/distribution-{self.distribution_type}_seedNum-{m}.pkl"

    def deliverers_cache_file(self, method, m = 0):
        return f"{self.data_prefix}/{self.data_set}/seeds-with-{self.data_set}/distribution-{self.distribution_type}_{method}_seedNum-{m}.txt"


    def usage_rate_file(self, m = 0):
        times = ",".join(str(time) for time in self.simulation_times)
        # todo 后续改为追加文件
        if self.distribution_type == 'powerlaw':
            return f"{self.data_prefix}/{self.data_set}/E-activated-{self.data_set}/distribution-{self.distribution_type}/tsd_{self.tran_degree_influence_factor}-{self.succ_degree_influence_factor}-{self.dis_degree_influence_factor}/simuTimes-{times}_seedNum-{m}_monteCarloL-{self.monte_carlo_L}_rrNumSamples-{self.num_samples}.csv"
        return f"{self.data_prefix}/{self.data_set}/E-activated-{self.data_set}/distribution-{self.distribution_type}_simuTimes-{times}_seedNum-{m}_monteCarloL-{self.monte_carlo_L}_rrNumSamples-{self.num_samples}.csv"


    def log_file(self):
        times = ",".join(str(time) for time in self.simulation_times)
        if self.distribution_type == "powerlaw" or self.distribution_type == "gamma" or self.distribution_type == "poisson":
            return f"/home/wen/pythonspace/coup_spread/gzc-impl/logs/{self.data_set}/distribution-{self.distribution_type}_seedNum-{self.seeds_num}_simuTimes-{times}_t-s-d-baseValue-{self.tran_base_value, self.succ_base_value, self.dis_base_value}_t-s-d-factor-{self.tran_degree_influence_factor, self.succ_degree_influence_factor, self.dis_degree_influence_factor}.log"
        return f"/home/wen/pythonspace/coup_spread/gzc-impl/logs/{self.data_set}/distribution-{self.distribution_type}_seedNum-{self.seeds_num}_simuTimes-{times}.log"