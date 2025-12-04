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
    succ_base_value: float = 1.0
    succ_degree_influence_factor: float = -0.1
    dis_base_value: float = 1.0
    dis_degree_influence_factor: float = -0.3
    tran_base_value: float = 1.0
    tran_degree_influence_factor: float = 0.5
    randomness_factor: float = 0.2

    rng: np.random.Generator = np.random.default_rng(1)

    single_sim_func: str = 'AgainContinue' # AgainContinue | AgainReJudge

    version: str = 'v0'

    random_dirichlet: list = None

    single_file_switch: bool = False

    
    @property
    def adj_file(self):
        return f"{self.data_prefix}/datasets/{self.data_set}-adj.pkl"

    # 当种子数和分布参数相同时 采用相同的概率分布
    def distribution_file(self, m = 0):
        return f"{self.data_prefix}/{self.data_set}/{self.version}/distribution-in-{self.data_set}/distribution-{self.distribution_type}_seedNum-{m}.pkl"

    def deliverers_cache_file(self, method, m = 0):
        return f"{self.data_prefix}/{self.data_set}/{self.version}/seeds-with-{self.data_set}/{self.distribution_type}_{method}_seedNum-{m}_SSRNum-{self.num_samples}.txt"


    def usage_rate_file(self, m = 0):
        times = ",".join(str(time) for time in self.simulation_times)
        if self.distribution_type == 'powerlaw':
            if self.single_file_switch:
                return f"{self.data_prefix}/{self.data_set}/{self.version}/E-activated-{self.data_set}/distribution-{self.distribution_type}/tsd_{self.tran_degree_influence_factor}-{self.succ_degree_influence_factor}-{self.dis_degree_influence_factor}/single_sim_func-{self.single_sim_func}/simuTimes-{times}_seedNum-{m}_monteCarloL-{self.monte_carlo_L}_single_sim_func-{self.single_sim_func}_rrNumSamples-{self.num_samples}.csv"
            else:
                return f"{self.data_prefix}/{self.data_set}/{self.version}/E-activated-{self.data_set}/distribution-{self.distribution_type}/tsd_{self.tran_degree_influence_factor}-{self.succ_degree_influence_factor}-{self.dis_degree_influence_factor}/single_sim_func-{self.single_sim_func}/simuTimes-{times}_monteCarloL-{self.monte_carlo_L}_single_sim_func-{self.single_sim_func}_rrNumSamples-{self.num_samples}.csv"
        return f"{self.data_prefix}/{self.data_set}/{self.version}/E-activated-{self.data_set}/distribution-{self.distribution_type}_simuTimes-{times}_seedNum-{m}_monteCarloL-{self.monte_carlo_L}_rrNumSamples-{self.num_samples}.csv"


    def log_file(self):
        times = ",".join(str(time) for time in self.simulation_times)
        if self.distribution_type == "powerlaw" or self.distribution_type == "gamma" or self.distribution_type == "poisson":
            return f"/home/wen/pythonspace/coup_spread/gzc-impl/logs/{self.data_set}/{self.version}/{self.distribution_type}_SSRNum-{self.num_samples}-_seedNum-{self.seeds_num}_simuTimes-{times}_t-s-d-baseValue-{self.tran_base_value, self.succ_base_value, self.dis_base_value}_t-s-d-factor-{self.tran_degree_influence_factor, self.succ_degree_influence_factor, self.dis_degree_influence_factor}.log"
        return f"/home/wen/pythonspace/coup_spread/gzc-impl/logs/{self.data_set}/{self.version}/{self.distribution_type}_SSRNum-{self.num_samples}-_seedNum-{self.seeds_num}_simuTimes-{times}.log"