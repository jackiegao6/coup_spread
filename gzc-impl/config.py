from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    data_set: str #数据集名字 
    simulation_times: list #控制评估的精度（模拟次数）
    methods: list = None #选投放点的方法
    seed_num_list: list = None #选的初始投放点
    monte_carlo_L: int = 10 # 蒙特卡洛模拟次数
    distribution: str = 'random' # 随机、泊松、正态三种
    constant_factor_distri: str = 'random' # todo random??
    personalization: str = 'None'
    data_prefix: str = '/root/autodl-tmp/data-processed'
    method_type: str = 'None'

    
    @property
    def adj_file(self):
        return f"{self.data_prefix}/{self.data_set}-adj.pkl"

    @property
    def distribution_file(self):
        # m = self.seed_num_list[-1]
        m = 10
        return f"{self.data_prefix}/distribution-{self.data_set}/distri-{self.distribution}_constantFactor{self.constant_factor_distri}_seedNum{m}.pkl"

    @property
    def deliverers_cache_file(self):
        # m = self.seed_num_list[-1]
        m = 10

        return f"{self.data_prefix}/dataset-{self.data_set}/deliverers_{self.data_set}_distri{self.distribution}_constantFactor{self.constant_factor_distri}_monteCarloL{self.monte_carlo_L}_seedNum{m}.txt"

    @property
    def usage_rate_file(self):
        # m = self.seed_num_list[-1]
        m = 10

        return f"{self.data_prefix}/dataset-{self.data_set}/usageRate_{self.data_set}_distri{self.distribution}_constantFactor{self.constant_factor_distri}_monteCarloL{self.monte_carlo_L}_testTimes{self.simulation_times}_seedNum{m}_{self.personalization}.txt"