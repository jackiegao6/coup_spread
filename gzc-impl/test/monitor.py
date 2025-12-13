import psutil, time, os
p = psutil.Process(os.getpid())

while True:
    print(f"CPU%: {p.cpu_percent():.1f}  "
          f"内存: {p.memory_info().rss/1024**2:.0f}MB  "
          f"线程数: {p.num_threads()}  "
          f"IO读写: {p.io_counters().read_bytes/1024**2:.1f}MB / {p.io_counters().write_bytes/1024**2:.1f}MB")
    time.sleep(1)