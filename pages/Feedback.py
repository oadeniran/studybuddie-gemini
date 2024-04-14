import streamlit as st
import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()
from utils import log_activity


base_url = os.getenv("BaseUrl")

st.title('StudyBuddy Feedback and Reviews😎😎')

if 'loggedIn' in st.session_state:
        log_activity("visit-feedback-page")
        st.write("✨✨Thanks for trying out studdy buddy. Kindly drop your reviews and feedbacks✨✨")

        st.write("We expect feedback on things you found off or things you would like to be implemented")

        feedback = st.text_area("Your Feedback", height=50)

        submit_button = st.button("!SUBMIT!")

        if submit_button:
            st.info("Submitting feedback")
            log_activity("submitted-feedback")
            if len(feedback) > 5:
                payload = {
                "user_id" : st.session_state['loggedIn']["uid"],
                "feedback": feedback
                }
                url = f"{base_url}/post-feedback"
                result = requests.post(
                    url,
                    data =payload,
                )
                contents_of_results = result.content
                contents_of_results = json.loads(contents_of_results.decode('utf-8'))
                if contents_of_results['status_code'] == 200:
                    st.success(contents_of_results['message'])
                    log_activity("finished-feedback")
                else:
                    st.error(contents_of_results['message'])
            else:
                st.error("You must provide a feedback")

else:
    st.write("Please login to give reviews")
