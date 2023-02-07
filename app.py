# import streamlit as st

# st.title('ndn test')
# st.write('test')

import pulp as pl
solver_list = pl.listSolvers()
print(solver_list)