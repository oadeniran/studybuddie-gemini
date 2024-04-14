import os
import random
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
import requests
import json
from utils import load_cont, load_blob, bot_from_load, load_VS_from_azure
import time

load_dotenv()
base_url = os.getenv("BaseUrl")

def signUp():
    st.title("Sign Up")
	
    with st.form("signup",clear_on_submit=False):
        username = st.text_input("Username (lowercase)")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        level = st.selectbox("What is your level", ["College", "Highschool", "Grade school"])
        college_n = st.text_input("University name")
        signUp_button = st.form_submit_button('Sign Up')

    required_fields = [
        (username, "Username field is required."),
        (password, "Password field is required."),
        (password, "Password field is required.")
    ]
	
    if signUp_button:
        errors = [error_msg for field, error_msg in required_fields if not field]
        if errors:
            for error in errors:
                st.error(error)
        else:
            payload = {
                "username" : username.lower(),
                "email" : email,
                "password" : password,
                "level" : level,
                "college_name": college_n
            }
            url = f"{base_url}/signup"
            result = requests.post(
                url,
                data =payload,
            )
            contents_of_results = result.content
            contents_of_results = json.loads(contents_of_results.decode('utf-8'))
            if contents_of_results['status_code'] == 200:
                st.success(contents_of_results['message'])
                st.info("Now proceed to sign in")
            else:
                st.error(contents_of_results['message'])

def login():
    st.title("Login")

    with st.form("login",clear_on_submit=True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button('login')

    required_fields = [
    (username, "Username field is required."),
    (password, "Password field is required."),
    ]
    if login_button:
        errors = [error_msg for field, error_msg in required_fields if not field]
        if errors:
            for error in errors:
                st.error(error)
        else:
            payload = {
                "username" : username.lower(),
                "password" : password
            }
            url = f"{base_url}/login"
            result = requests.post(
                url,
                data=payload,
            )
            contents_of_results = result.content
            contents_of_results = json.loads(contents_of_results.decode('utf-8'))
            if contents_of_results['status_code'] == 200:
                st.success(contents_of_results['message'])
                st.info("Processing previous categories---------")
                try:
                    st.session_state['loggedIn'] = {"uid" : contents_of_results["uid"]}
                    cat_res = requests.post(f"{base_url}/get-user-categories", data = {"uid" : contents_of_results["uid"]},)
                    cat_res = cat_res.content
                    cat_res = json.loads(cat_res.decode('utf-8'))
                    if cat_res["status_code"] == 200:
                        st.session_state["categories"] = cat_res["categories"]
                        st.session_state["category_det"] = cat_res["category_det"]
                        st.session_state["categories_dict"] = cat_res["categories_dict"]
                        st.session_state["bots"] ={}
                        st.session_state["history_dict"] = cat_res["history_dict"]
                        cont = load_cont(st.session_state['loggedIn']["uid"])
                        for cat_name in st.session_state["categories_dict"].keys():
                            st.session_state["bots"][cat_name[4:]]= [bot_from_load(load_VS_from_azure(cont, blob_name=v['bot']), pdf_name=k) for k, v in st.session_state["categories_dict"][cat_name].items()]
                        
                        st.session_state["activities"], st.session_state['start_time']  = [f"loggedin-{round(time.time(),2)}"], round(time.time(),2)
                        st.session_state["uid+date"] =    st.session_state['loggedIn']["uid"] + '-'+time.strftime('%x') 
                        st.switch_page("pages/My-Categories.py")
                    else:
                        st.session_state["history_dict"] = {}
                        st.session_state["bots"] ={}
                        st.info("No previous categories found")

                    st.session_state["activities"], st.session_state['start_time']  = [f"loggedin-{round(time.time(),2)}"], round(time.time(),2)
                    st.session_state["uid+date"] =    st.session_state['loggedIn']["uid"] + '-'+time.strftime('%x') 
                    st.info("...Successfully signed in. You can use all features now...")
                    
                except Exception as e:
                    print(e)
                    st.error("Error in processing categories")
                    st.info("Please try again or contatct support. Thanks!")
                    del st.session_state['loggedIn']
            else:
                st.error(contents_of_results['message'])


def main():
    if 'loggedIn' in st.session_state:
        st.write("The Onestop AI tool to help you power through your study. Kindly check the about page to see all details and features.")

        st.write("You are logged in, proceed to use all our wonderful features")
    else:
        st.title('Welcome to StudyBuddy😎😎')

        # Adding a brief description
        st.write("""Being your StudyBuddy, my Job is to help you power through your study with ease. Kindly Signup to access all features if you do not have an account. If you have an account already, please Login to proceed.""")

        st.sidebar.title("Option")

        selection = st.sidebar.radio("Go to", ["Sign up","Login"])
        if selection == "Sign up":
            signUp()
        elif selection == "Login":
            login()


if __name__ == "__main__":
    main()
