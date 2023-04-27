import numpy as np
import pulp
import streamlit as st
import json
import pandas as pd
st.set_page_config(layout="wide")
model_data = {}

tab_load, tab_processes, tab_workers, tab_model, tab_save = st.tabs(
    ['load from file', 'processes', 'workers', 'model', 'save to file'])

with tab_load:
    uploaded_file = st.file_uploader(label='load parameters from file', type=['json'])
    if uploaded_file is not None:
        model_data = json.loads(uploaded_file.read())
        print('data loaded')
        print(model_data)

with tab_processes:
    col_volumes, col_normatives, col_table = st.columns(3)
    col_volumes.title('Volumes')
    col_normatives.title('Normatives')
    col_table.title('working volumes')

    model_data['num_processes'] = st.slider(label='number of processes', min_value=1, max_value=30, value=model_data.get('num_processes', 1))

    with col_volumes:
        model_data['process_volumes'] = [0] * model_data['num_processes']
        for i in range(model_data['num_processes']):
            model_data['process_volumes'][i] = st.number_input(
                f'volume for process {i}:', min_value=0, value=model_data['process_volumes'][i], step=1)

    with col_normatives:  
        model_data['process_normatives'] = [0] * model_data['num_processes']
        for i in range(model_data['num_processes']):
            model_data['process_normatives'][i] = st.number_input(
                f'normative for process {i}:', min_value=0, max_value=30, value=model_data['process_normatives'][i] if model_data['process_normatives'][i] > 0 else 1, step=1)

    with col_table:
        volume_table = pd.DataFrame([
            model_data['process_volumes'], 
            model_data['process_normatives'], 
            np.array(model_data['process_volumes']) * np.array(model_data['process_normatives']),
            np.array(model_data['process_volumes']) * np.array(model_data['process_normatives']) / 480,
            ]).transpose()
        volume_table.columns = ['volume', 'normative', 'minutes', 'FTE']
        st.table(volume_table)            

with tab_workers:
    model_data['num_workers'] = st.slider('number of workers', 1, 30)
    worker_skills = [0] * model_data['num_workers']
    for i in range(model_data['num_workers']):
        worker_skills[i] = st.multiselect(f'worker {i} skills:',
                                                options=[j for j in range(model_data['num_processes'])])
    model_data['worker_skills_matrix'] = []
    for i in range(model_data['num_workers']):
        a = [0] * model_data['num_processes']
        for skill in worker_skills[i]:
            a[skill] = 1
        model_data['worker_skills_matrix'].append(a.copy())
    st.table(np.array(model_data['worker_skills_matrix']))

with tab_model:
    model = pulp.LpProblem('workers to processes', pulp.LpMinimize)

    is_working = pulp.LpVariable.dicts(
        'is_working', range(model_data['num_workers']), cat='Binary')

    model += pulp.lpSum(is_working)

    items_worker_process = pulp.LpVariable.dicts(
        'items_worker_process', (range(model_data['num_workers']), range(model_data['num_processes'])), lowBound=0, cat=pulp.LpInteger)
    
    worker_start = pulp.LpVariable.dicts(
        'worker_start', range(model_data['num_workers']), lowBound=0, upBound=23, cat=pulp.LpInteger)
    worker_end = pulp.LpVariable.dicts(
        'worker_end', range(model_data['num_workers']), lowBound=0, upBound=24, cat=pulp.LpInteger)
    
    worker_work_minutes = pulp.LpVariable.dicts(
        'worker_work_minutes', range(model_data['num_workers']), lowBound=0, cat=pulp.LpInteger)

    for worker in range(model_data['num_workers']):
        worker_work_minutes[worker] = (worker_end[worker] - worker_start[worker]) * 60
        model += (worker_work_minutes[worker] <= 480, f'worker {worker} works no more than 480 minutes (shiftly)')
    
    for process in range(model_data['num_processes']):
        model += (
            pulp.LpAffineExpression(
                [(items_worker_process[worker][process], model_data['worker_skills_matrix'][worker][process])
                for worker in range(model_data['num_workers'])]) == model_data['process_volumes'][process],
            f'process {process} to be done completely'
        )

    for worker in range(model_data['num_workers']):
        model += (
            pulp.LpAffineExpression([(items_worker_process[worker][process], model_data['worker_skills_matrix'][worker][process] * model_data['process_normatives'][process])
                                    for process in range(model_data['num_processes'])]) <= worker_work_minutes[worker],
            f'worker {worker} works from start till end')

    for worker in range(model_data['num_workers']):
        model += (
            worker_work_minutes[worker] / 480 <= is_working[worker],
            f'worker {worker} works if he has minutes to work'
        )

    with st.expander("model description"):
        st.text(model)

    status = model.solve()
    if status == pulp.LpStatusOptimal:
        st.write('working workers:')
        workers_str = ""
        for worker in range(model_data['num_workers']):
            if is_working[worker].value() > 0:
                workers_str += str(worker) + ' '
        st.write(workers_str)
        st.write('production volumes:')
        worker_array = np.zeros((model_data['num_workers'], model_data['num_processes']))
        for worker in range(model_data['num_workers']):
            for process in range(model_data['num_processes']):
                worker_array[worker,
                            process] = int(items_worker_process[worker][process].value())
        st.table(worker_array)
        st.write('shift time:')
        shift_array = np.zeros((model_data['num_workers'], 3))
        for worker in range(model_data['num_workers']):
            shift_array[worker, 0] = int(worker_start[worker].value())
            shift_array[worker, 1] = int(worker_end[worker].value())
            shift_array[worker, 2] = int(worker_work_minutes[worker].value())
        shift_array = pd.DataFrame(shift_array, columns=['shift start', 'shift end', 'working minues'])
        st.table(shift_array)
    else:
        st.write('no solution')

with tab_save:
    st.json(model_data)
    st.download_button(
        label="Download parameters as JSON",
        data=json.dumps(model_data),
        file_name='test.json',
        mime='application/json',
    )
