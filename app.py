import streamlit as st
from user_query import main as user_query_main
from data_backend import main as data_backend_main

# Sidebar navigation
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", ['User Query', 'Data Backend'])

if selection == 'User Query':
    user_query_main()
elif selection == 'Data Backend':
    data_backend_main()
