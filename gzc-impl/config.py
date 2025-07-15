from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    data_set: str #数据集名字 
    simulation_times: list #控制评估的精度（模拟次数）
    methods: list = None #选种子集的方法
    seed_num_list: list = None #种子集列表
    monte_carlo_L: int = 10 # 蒙特卡洛模拟次数
    distribution_type: str = 'random' # 随机、泊松、正态三种
    constant_factor_distri: str = 'random' # todo random??
    personalization: str = 'None'
    data_prefix: str = '/root/autodl-tmp/processed-data'
    method_type: str = 'None'

    
    @property
    def adj_file(self):
        return f"{self.data_prefix}/datasets/{self.data_set}-adj.pkl"

    def distribution_file(self, m = -1):
        # m = self.seed_num_list[-1]
        return f"{self.data_prefix}/{self.data_set}/distribution-{self.data_set}/distri-{self.distribution_type}_constantFactor{self.constant_factor_distri}_seedNum{m}.pkl"

    def deliverers_cache_file(self, m = -1):
        # m = self.seed_num_list[-1]
        return f"{self.data_prefix}/{self.data_set}/deliverers_{self.data_set}_distri{self.distribution_type}_constantFactor{self.constant_factor_distri}_monteCarloL{self.monte_carlo_L}_seedNum{m}.txt"

    def usage_rate_file(self, m = -1):
        # m = self.seed_num_list[-1]
        return f"{self.data_prefix}/{self.data_set}/usageRate_{self.data_set}_distri{self.distribution_type}_constantFactor{self.constant_factor_distri}_monteCarloL{self.monte_carlo_L}_testTimes{self.simulation_times}_seedNum{m}_{self.personalization}.txt"