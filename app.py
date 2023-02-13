import numpy
import pulp
import streamlit as st

tab_processes, tab_workers, tab_model = st.tabs(
    ['processes', 'workers', 'model'])

num_processes = tab_processes.slider('number of processes', 1, 30)

col_volumes, col_normatives = tab_processes.columns(2)

# num_processes = 3
col_volumes.title('Volumes')
process_volumes = [0] * num_processes
for i in range(num_processes):
    process_volumes[i] = col_volumes.number_input(
        f'volume for process {i}:', min_value=0, value=process_volumes[i], step=1)
# process_volumes = [10, 20, 30]
col_normatives.title('Normatives')
process_normatives = [0] * num_processes
for i in range(num_processes):
    process_normatives[i] = col_normatives.number_input(
        f'normative for process {i}:', min_value=0, max_value=20, value=process_normatives[i], step=1)

# process_normatives = [10, 20, 20]

# num_workers = 7
num_workers = tab_workers.slider('number of workers', 1, 30)
worker_skills = [0] * num_workers
for i in range(num_workers):
    worker_skills[i] = tab_workers.multiselect(f'worker {i} skills:',
                                               options=[j for j in range(num_processes)])
# worker_skills = [
#     [0],
#     [1],
#     [2],
#     [],
#     [],
#     [],
#     []
# ]

worker_skills_matrix = []
for i in range(num_workers):
    a = [0] * num_processes
    for skill in worker_skills[i]:
        a[skill] = 1
#    a = list(numpy.multiply(a, process_normatives))
    worker_skills_matrix.append(a.copy())
tab_workers.table(numpy.array(worker_skills_matrix))

model = pulp.LpProblem('workers to processes', pulp.LpMinimize)

is_working = pulp.LpVariable.dicts(
    'is_working', range(num_workers), cat='Binary')

model += pulp.lpSum(is_working)

items_worker_process = pulp.LpVariable.dicts(
    'items_worker_process', (range(num_workers), range(num_processes)), lowBound=0, cat=pulp.LpInteger)

for process in range(num_processes):
    model += (
        pulp.LpAffineExpression(
            [(items_worker_process[worker][process], worker_skills_matrix[worker][process])
             for worker in range(num_workers)]) == process_volumes[process],
        f'process {process} to be done completely'
    )

for worker in range(num_workers):
    model += (
        pulp.LpAffineExpression([(items_worker_process[worker][process], worker_skills_matrix[worker][process] * process_normatives[process])
                                for process in range(num_processes)]) - 480 * is_working[worker] <= 0,
        f'worker {worker} works maximum 480 minutes per day')

for worker in range(num_workers):
    model += (
        pulp.lpSum([items_worker_process[worker][process]
                    for process in range(num_processes)]) / (sum(process_volumes) if sum(process_volumes) > 0 else 1) <= is_working[worker],
        f'worker {worker} works if he has tasks'
    )

# print(model)

status = model.solve()
if status == pulp.LpStatusOptimal:
    tab_model.write('working workers:')
    workers_str = ""
    for worker in range(num_workers):
        if is_working[worker].value() > 0:
            workers_str += str(worker) + ' '
    tab_model.write(workers_str)
    worker_array = numpy.zeros((num_workers, num_processes))
    for worker in range(num_workers):
        for process in range(num_processes):
            worker_array[worker,
                         process] = int(items_worker_process[worker][process].value())
    tab_model.table(worker_array)
else:
    tab_model.write('no solution')
