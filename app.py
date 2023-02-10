# import streamlit as st

# st.title('ndn test')
# st.write('test')

import pulp
import numpy
# solver_list = pulp.listSolvers()
# print(solver_list)

num_processes = 3
num_workers = 7

process_volumes = [10, 20, 30]

MAX_VOLUME = -1
for v in process_volumes:
    if v > MAX_VOLUME:
        MAX_VOLUME = v

MAX_VOLUME = 1000

process_normatives = [10, 20, 15]

process_volumes_minutes = [1] * num_processes
for i in range(num_processes):
    process_volumes_minutes[i] = process_volumes[i] * process_normatives[i]

worker_skills = [
    [0],
    [1],
    [2],
    [],
    [],
    [],
    []
]

worker_skills_matrix = []
for i in range(num_workers):
    a = [0] * num_processes
    for skill in worker_skills[i]:
        a[skill] = 1
    a = list(numpy.multiply(a, process_normatives))
    worker_skills_matrix.append(a.copy())
print(worker_skills_matrix)

model = pulp.LpProblem('workers to processes', pulp.LpMinimize)

is_working = pulp.LpVariable.dicts(
    'is_working', range(num_workers), cat='Binary')

model += pulp.lpSum(is_working)

items_worker_process = pulp.LpVariable.dicts(
    'items_worker_process', (range(num_workers), range(num_processes)), lowBound=0, cat=pulp.LpInteger)

# do all items in process:
for process in range(num_processes):
    model += (
        pulp.LpAffineExpression(
            [(items_worker_process[worker][process], worker_skills_matrix[worker][process])
             for worker in range(num_workers)]) >= process_volumes_minutes[process],
        f'process {process} to be done completely'
    )

# work max 480 minutes per day if works
for worker in range(num_workers):
    model += (
        pulp.LpAffineExpression([(items_worker_process[worker][process], worker_skills_matrix[worker][process])
                                for process in range(num_processes)]) - 480 * is_working[worker] <= 0,
        f'worker {worker} works maximum 480 minutes per day')

for worker in range(num_workers):
    model += (
        pulp.lpSum([items_worker_process[worker][process]
                    for process in range(num_processes)]) / MAX_VOLUME <= is_working[worker],
        f'worker {worker} works if he has tasks'
    )

# how to remove workers ~is_working

print(model)

status = model.solve()
if status == pulp.LpStatusOptimal:
    for worker in range(num_workers):
        print(f'worker {worker} is working: {is_working[worker].value()}')
    for worker in range(num_workers):
        print(f'worker {worker}')
        for process in range(num_processes):
            print(
                f'{process}: {items_worker_process[worker][process].value()}', end=' ')
            print()
else:
    print('no solution')
