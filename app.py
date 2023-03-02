import numpy as np
import pulp
import pandas as pd
import random

MAX_MINUTES_PER_DAY = 8 * 60

STREAM_LIT_ON = False
if STREAM_LIT_ON:
    import streamlit as st

    st.set_page_config(layout="wide")

    tab_processes_volumes, tab_processes_normatives, tab_workers, tab_model = st.columns(4)

if STREAM_LIT_ON:
    num_processes = tab_processes_volumes.slider('number of processes', 1, 30)
else:
    num_processes = int(input('number of processes: '))

process_volumes = [0] * num_processes
if STREAM_LIT_ON:
    tab_processes_volumes.header('volumes')
    for i in range(num_processes):
        process_volumes[i] = tab_processes_volumes.number_input(
            f'volume for process {i}:', min_value=0, value=process_volumes[i], step=1)
else:
    process_volumes = np.random.randint(low=100, high=200, size=num_processes)

process_normatives = [0] * num_processes
if STREAM_LIT_ON:
    tab_processes_normatives.subheader('Normatives')
    for i in range(num_processes):
        process_normatives[i] = tab_processes_normatives.number_input(
            f'normative for process {i}:', min_value=0, max_value=MAX_MINUTES_PER_DAY, value=process_normatives[i], step=1)
else:
    process_normatives = np.random.randint(low=10, high=20, size=num_processes)

processes_matrix = pd.DataFrame(
    [process_volumes, process_normatives, [process_volumes[i] * process_normatives[i]
                                           for i in range(num_processes)], [process_volumes[i] * process_normatives[i] / MAX_MINUTES_PER_DAY
                                                                            for i in range(num_processes)]], index=['volume', 'normative', 'minutes', 'FTE'])
if STREAM_LIT_ON:
    tab_processes_volumes.dataframe(processes_matrix)
else:
    print(processes_matrix)

if STREAM_LIT_ON:
    num_workers = tab_workers.slider('number of workers', 1, 30)
else:
    num_workers = int(input('number of workers: '))

worker_skills = [0] * num_workers
if STREAM_LIT_ON:
    for i in range(num_workers):
        worker_skills[i] = tab_workers.multiselect(f'worker {i} skills:',
                                                options=[j for j in range(num_processes)])
else:
    for i in range(num_workers):
        worker_skills[i] = np.random.randint(low=0, high=num_processes, size=random(1, 5))

worker_skills_matrix = []
for i in range(num_workers):
    a = [0] * num_processes
    for skill in worker_skills[i]:
        a[skill] = 1
    worker_skills_matrix.append(a.copy())

if STREAM_LIT_ON:
    tab_workers.table(np.array(worker_skills_matrix))
else:
    print(np.array(worker_skills_matrix))

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
                                for process in range(num_processes)]) - MAX_MINUTES_PER_DAY * is_working[worker] <= 0,
        f'worker {worker} works maximum 480 minutes per day')

for worker in range(num_workers):
    model += (
        pulp.lpSum([items_worker_process[worker][process]
                    for process in range(num_processes)]) / (sum(process_volumes) if sum(process_volumes) > 0 else 1) <= is_working[worker],
        f'worker {worker} works if he has tasks'
    )

if STREAM_LIT_ON:
    with tab_model.expander("model description"):
        st.text(model)
else:
    print(model)

status = model.solve()
if status == pulp.LpStatusOptimal:
    workers_str = ""
    for worker in range(num_workers):
        if is_working[worker].value() > 0:
            workers_str += str(worker) + ' '
    if STREAM_LIT_ON:
        tab_model.write('working workers:')
        tab_model.write(workers_str)
    else:
        print(f'working workers: {workers_str}')

    worker_array = np.zeros((num_workers, num_processes))
    for worker in range(num_workers):
        for process in range(num_processes):
            worker_array[worker,
                         process] = int(items_worker_process[worker][process].value())
    if STREAM_LIT_ON:
        tab_model.table(worker_array)
    else:
        print(worker_array)
else:
    if STREAM_LIT_ON:
        tab_model.write('no solution')
    else:
        print('no solution')
