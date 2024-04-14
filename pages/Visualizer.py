import streamlit as st
from utils import log_activity

if 'loggedIn' not in st.session_state:
        st.error("Please Login to Use feature......Return Home to login")
else:
        log_activity("visit-visualize-page")
        st.header("Feature coming soon.....")