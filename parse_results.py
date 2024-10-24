import os
import re
import shutil
import numpy as np
import pandas as pd
from random import randint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


shutil.rmtree('graphs_tables/')
os.makedirs('graphs_tables', exist_ok=True)
N = [128, 512, 2048]
T = [1, 4, 16, 64]


##############
# Parsers 
##############

def parse_rt(
    method: str,
    N: list[int],
    T: list[int] = None
) -> dict[int, dict[int, float]]:
  results = {}
  for t in T:
    results[t] = {}
    for n in N:
      df = pd.read_csv(f"results/{method}_{n}_{t}_rt", sep=' ', skiprows=2, names=['n', 'runtime'])
      run_time = df[df['n'] == n]['runtime'].values[0]
      results[t][n] = run_time
  return results

perf_counter_fname = {
  'L2CACHE': 'L2CACHE',
  'L3CACHE': 'L3CACHE',
  'L3MISS': 'L3CACHE',
  'FLOPS_DP': 'FLOPS_DP',
}

perf_counter_dict = {
  'L2CACHE': 'REQUESTS_TO_L2_GRP1_ALL_NO_PF',
  'L3CACHE': 'L3_CACHE_REQ',
  'L3MISS': 'L3_CACHE_REQ_MISS',
  'FLOPS_DP': 'RETIRED_INSTRUCTIONS',
}

def get_perf_counter(method: str, n: int, perf_counter: str) -> int:
  perf_fname = perf_counter_fname[perf_counter]
  with open(f'./results/{method}_{n}_{perf_fname}', 'r') as f:
    content = f.read()
  metric_name = perf_counter_dict[perf_counter]
  match = re.search(rf'\|\s+{metric_name}\s+\|\s+\w+\s+\|\s+(\d+)\s+\|', content)
  if match:
    return int(match.group(1))
  else:
    raise ValueError(f"performance counter {metric_name} not found for method {method} n {n}")

##############
# Plotters 
##############

def get_n_colors(N):
  if N > len(mcolors.TABLEAU_COLORS):
    return ['#%06X' % randint(0, 0xFFFFFF) for _ in range(N)]
  return list(mcolors.TABLEAU_COLORS.keys())[:N]

marker_dict = {
  'basic':  'o',
  'strassen-64': 'p',
  'strassen-256': '*',
  'strassen-512': 'x',
  'blocked-4': '^',
  'blocked-16': 'v',
  'blas': 's'
}

def plot_runtimes(
  method_thread_list: list[tuple[str, int]], 
  figname: str = 'runtimes',
  mode: str = 'Runtime (s)'
):
  colors = get_n_colors(len(method_thread_list))
  for idx, (method, threads_num) in enumerate(method_thread_list):
    results = parse_rt(method, N, [threads_num])[threads_num]
    if mode == 'Runtime (s)':
      results = [results[n] for n in N]
    elif mode == 'FLOPs':
      results = [2 * n ** 3 / results[n] / 1e9 for n in N]
    elif mode == 'Speedup':
      results_1 = parse_rt(method, N, [1])[1]
      results = [results_1[n] / results[n] for n in N]
    else:
      raise ValueError(f"Unknown mode {mode}")
    plt.scatter(N, results, label=f"{method}, {threads_num} threads", marker=marker_dict[method], color=colors[idx])
    plt.plot(N, results, color=colors[idx])
  plt.xlabel('N')
  plt.ylabel(f'{mode}')
  plt.title(f'{mode} vs N')
  plt.legend()
  plt.savefig(f'graphs_tables/{figname}.png', dpi=300)
  plt.close()

def gen_perf_counter_table(methods: list[str], perf_counter: str):
  result_table = {}
  for n in N:
    result_table[n] = {}
    for method in methods:
      result = get_perf_counter(method, n, perf_counter)
      blas_result = get_perf_counter('blas', n, perf_counter)
      result_table[n][method] = result / blas_result
  pd.DataFrame(result_table).T.to_latex(f'graphs_tables/{perf_counter}_tex_table', float_format=f"%.1f")

##############
# Main
##############

def main():
  gen_perf_counter_table(['basic', 'blocked-4', 'blocked-16'], 'L2CACHE')
  gen_perf_counter_table(['basic', 'blocked-4', 'blocked-16'], 'L3CACHE')
  gen_perf_counter_table(['basic', 'blocked-4', 'blocked-16'], 'L3MISS')
  gen_perf_counter_table(['basic', 'blocked-4', 'blocked-16'], 'FLOPS_DP')

  threaded_methods = ['basic'] + [f'strassen-{l}' for l in (64, 256, 512)] + [f'blocked-{b}' for b in (4, 16)]
  for t in T:
    for mode in ['Runtime (s)', 'FLOPs']:
      plot_runtimes(
        [('blas', 1)] + [(method, t) for method in threaded_methods], 
        f'{mode}_{t}threads', mode)

  threaded_methods = ['basic'], [f'strassen-{l}' for l in (64, 256, 512)], [f'blocked-{b}' for b in (4, 16)]
  for methods in threaded_methods:
    plot_runtimes([(method, t) for t in T for method in methods], 
                  f'speedup_{methods[0].split("-")[0]}', 'Speedup')


if __name__ == "__main__":
  main()