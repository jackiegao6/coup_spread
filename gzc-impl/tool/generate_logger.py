import logging
import sys
def init_logger(log_file='logs/eval.log'):
    logger = logging.getLogger()  # 获取 root logger
    logger.setLevel(logging.INFO)

    # 防止多次添加 handler（重复输出）
    if logger.hasHandlers():
        return

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)