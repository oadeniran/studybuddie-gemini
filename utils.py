#some code was here
from azure.storage.blob import BlobServiceClient
import os
import streamlit as st
from dotenv import load_dotenv
import streamlit as st
import os
import re
import random
import traceback
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents.base import Document
import time
import json
import requests
import pandas as pd
import numpy as np
import google.generativeai as genai
import textwrap


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
blob_conn_str = os.getenv("BLOLB_CONNECTION_STRING")
OBJECTIVE="Objective"
TRUE_OR_FALSE="True/False"
FLASH_CARDS='Flash Cards😍'
OBJECTIVE_DEFAULT_VALUE=5
TRUE_OR_FALSE_DEFAULT_VALUE=5
base_url = os.getenv("BaseUrl")


def embed_fn(title, text):
  return genai.embed_content(model="models/embedding-001",
                             content=text,
                             task_type="retrieval_document",
                             title=title)["embedding"]

def find_best_passage(query, dataframe):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  query_embedding = genai.embed_content(model="models/embedding-001",
                                        content=query,
                                        task_type="retrieval_query")
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
  
  idxs = np.argpartition(dot_products, -4)[-4:]
  return '\\'.join(txt for txt in dataframe.iloc[idxs]['page_content']) # Return text from index with max value


def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""
    You are a helpful and informative bot that answers questions related to a pdf. 
    1. If the question is a greeting like "Hi" or a generic sentence like "How are you" and reply accordingly.
    2. However, if the question relates to the pdf then answer based on the following instructions
    A. The titled "PASSAGE OF PDF" below contains the most likely contents of the pdf that can be used to answer the "queston" \
    B. Be sure to respond in a complete sentence, being comprehensive, including all relevant information. \
    C. You are talking to a non-technical audience, so be sure to break down complicated concepts and \
        strike a friendly and converstional tone. \
    D. Ensure you say don't give answers that does not directly relate to the contents of passage of the pdf provided if answering a query about the pdf
  QUESTION: '{query}'
  PASSAGE OF PDF: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt

llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

llm = ChatOpenAI(    
        openai_api_key=OPENAI_API_KEY, 
        model_name="gpt-4", 
        temperature=0.0
    )

llm = ChatOpenAI(    
        openai_api_key=OPENAI_API_KEY, 
        model_name="gpt-4", 
        temperature=0.0
    )

class Agency:
    def __init__(self):
        self.pdf_agent=None
        self.tools = None
        self.agent = None
    
    def initialize_tools(self,pdf_agent,pdf_name):
        self.pdf_agent = pdf_agent
        tools = [
            Tool(
                name = "PDF Question and Answering Assistant",
                func=self.pdf_agent.run,
                description=f"""
                Useful for answering questions related to the uploaded pdf, the name of the pdf file is {pdf_name}
                """
            )
        ]
        self.tools = tools
    def load_agent_details(self,memory):
        agent_kwargs = {
            'prefix': f"""
            You are a friendly PDF Question and Answering Assistant.
            You are tasked to assist the current user on questions related to the uploaded PDF file.
            You have access to the following tools:{self.tools}.
             Try as much as possible to get your answers only from the tools provided to you. If you can not answer correctly, respond appropriately"""}
        # initialize the LLM agent
        agent = initialize_agent(self.tools, 
                                llm, 
                                agent="chat-conversational-react-description", 
                                verbose=True, 
                                agent_kwargs=agent_kwargs,
                                handle_parsing_errors=True,
                                memory=memory
                                )
        self.agent = agent
    def get_response(self,user_input):
        response = self.agent({"input":user_input,"chat_history":[]})
        return response


def log_activity(name):
    st.session_state['activities'].append((name + '-' + str(round(time.time() - st.session_state["start_time"], 2))))
    payload = json.dumps({
                    "uid_date": st.session_state["uid+date"],
                    "activity": st.session_state['activities']
                })
    url = f"{base_url}/update-user-activity"
    result = requests.post(
        url,
        headers={'Content-Type': 'application/json'},
        data=payload,
    )


def load_cont(container_name):
    blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
    container_client = blob_service_client.get_container_client(container_name)
    if not container_client.exists():
        container_client = container_client.create_container()
        container_client = blob_service_client.get_container_client(container_name)
    return container_client

def load_blob(container_client, blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client

def load_VS_from_azure(container_client, blob_name):
    bc = load_blob(container_client, blob_name)
    f = bc.download_blob()
    return f.read()

def bot_from_load(loaded_bytes, pdf_name):
    embeddings = OpenAIEmbeddings()
    db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=loaded_bytes) 
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agency = Agency()
    agency.initialize_tools(qa,pdf_name)
    agency.load_agent_details(memory)
    return agency


# Upload Section
def upload(uid, category_name, category_dict):
    st.header('Upload Your PDF')
    st.write("Drag and drop your PDF file here or click to upload. Please ensure that the text in the PDF is selectable and not a scanned image.")
    uploaded_file = st.file_uploader("", type="pdf")

    if uploaded_file is not None:
        # Check if the uploaded file is a PDF
        pdf = uploaded_file.name
        pdf_name = '.'.join(pdf.split('.')[:-1])
        with open(pdf, mode='wb') as f:
            f.write(uploaded_file.getbuffer()) # save pdf to disk
        st.success("Uploading File.....")
        loader = PyPDFLoader(pdf)
        category_dict[pdf_name] = {}
        category_dict[pdf_name]["pages"]= [d.dict() for d in loader.load_and_split()]
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=20)
        texts = text_splitter.split_documents(documents)
        if len(texts) == 0:
            st.error("Please ensure your uploaded  document is selectable (i.e not scanned)")
        else:
            st.success("File uploaded successfully!")
            st.write("Processing Uploaded PDF..........please wait till success message")
            # Create the dict for uploaded pdf
            embeddings = OpenAIEmbeddings()
            try:
                db = FAISS.from_documents(texts,embeddings)
                retriever = db.as_retriever()
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                )
                pkl = db.serialize_to_bytes()
                with open(f"my_file.txt", "wb") as binary_file:
                    # Write bytes to file
                    binary_file.write(pkl)
                cont = load_cont(uid)
                #print(cont)
                blob_client = load_blob(cont, blob_name=f"{category_name}_{pdf_name}")

                with open("my_file.txt", "rb") as f:
                    res = blob_client.upload_blob(f, overwrite=True)
                
                category_dict[pdf_name]["bot"] = f"{category_name}_{pdf_name}"
                
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                agency = Agency()
                agency.initialize_tools(qa,pdf_name)
                agency.load_agent_details(memory)
                if category_name not in st.session_state["bots"]:
                    st.session_state["bots"][category_name] = []
                st.session_state["bots"][category_name].append(agency)
                log_activity('upload-done')
                st.success("PDF processed Successfully!!!")
                
                return "done"
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()
                st.error("A network error occured, Please check your internet connection and try again")

def chatbot(pdf_name, category_name,ind, history_dict):
        st.header("StudyBud")
        qa = st.session_state['bots'][category_name][ind]
        st.markdown(f"Ask questions related to the uploaded file here : {pdf_name}")
        message("Hiiiiii!!, I am your studyBud, your personal reading buddy  😊 😊 😊!!!!")
        if "past" not in history_dict:
            history_dict['past'] = []
        if "generated" not in history_dict:
            history_dict["generated"] = []
        if "input_message_key" not in history_dict:
            history_dict["input_message_key"] = str(random.random())
        chat_container = st.container()
        user_input = st.text_area("Type your question here.", key=history_dict["input_message_key"])
        if st.button("Send"):
            log_activity('send-message')
            response = qa.get_response(user_input)
            history_dict["past"].append(user_input)
            history_dict["generated"].append(response['output'])
            history_dict["input_message_key"] = str(random.random())
            st.rerun()
        if history_dict["generated"]:
             with chat_container:
                  for i in range(len(history_dict["generated"])):
                    message(history_dict["past"][i], is_user=True, key=str(i) + "_user")
                    message(history_dict["generated"][i], key=str(i))

def parse_questions(input_text):
    question_blocks = input_text.split('QUESTION')

    questions = []

    for block in question_blocks:
        #print("BLOCK is", block )
        if block.strip() and "CHOICE_A" in block:
            
            question_num = int(re.search(r'\d+', block).group())

          
            question_text = re.search(r'(?<=:\s).*?(?=\n)', block).group()

         
            choices = re.findall(r'(CHOICE_[A-E]):\s(.*?)\n', block)
            options = {choice[0][-1]: choice[1] for choice in choices}

         
            answer = re.search(r'Answer:\s([A-E])', block).group(1)

        
            question_obj = {
                "question_num": question_num,
                "question": question_text,
                "options": options,
                "answer": answer
            }

          
            questions.append(question_obj)

    return questions

def generate_quizz(document,type,num_of_questions):
    if type == OBJECTIVE:
        

        prompt_template =  """You are a teacher preparing questions for a quiz. Given the following document, please generate {num} multiple-choice questions (MCQs) with 5 options and a corresponding answer letter based on the document except for CHOICE_A which should ALWAYS BE defaulted to NONE.please do not make any reference of the word "document"

            Example question:

            QUESTION 1: question here
            CHOICE_A: NONE
            CHOICE_B: choice here
            CHOICE_C: choice here
            CHOICE_D: choice here
            CHOICE_E: choice here
            Answer: C
            
            QUESTION 2: question here
            CHOICE_A: NONE
            CHOICE_B: choice here
            CHOICE_C: choice here
            CHOICE_D: choice here
            CHOICE_E: choice here
            Answer: B



            These questions should be detailed and solely based on the information provided in the document.

            <Begin Document>
            {doc}
            <End Document>"""
    elif type == TRUE_OR_FALSE:
        prompt_template =  """You are a teacher preparing True or False questions for a quiz. Given the following document, please generate {num} True or False type questions  with 3 options and a corresponding True or False answer except for CHOICE_A which should ALWAYS BE defaulted to NONE
            For each question, randomly select a section or sentence from the document, and then create a question that is directly based on the text. please do not make any reference of the word "document"

            Remember to vary the type of information each question is based on. Also, ensure each question has a different structure or focus to avoid repetition.


            Example question:

            QUESTION 1: question here
            CHOICE_A: NONE
            CHOICE_B: choice here
            CHOICE_C: choice here
            Answer: B

            QUESTION 2: question here
            CHOICE_A: NONE
            CHOICE_B: choice here
            CHOICE_C: choice here
            Answer: C



            These questions should be detailed and solely based on the information provided in the document.


            <Begin Document>
            {doc}
            <End Document>"""
    elif type == FLASH_CARDS:
        prompt_template =  """You are a teacher preparing Flash Cards type questions for a quiz. Given the following document, please generate {num} questions and a corresponding short answer 
            For each question, randomly select a section or sentence from the document, and then generate a useful question that would help your students in understanding the material .
            When generating the question, please do not make any reference of the word "document"

            Remember to vary the type of information each question is based on. Also, ensure each question has a different structure or focus to avoid repetition.


            Example question:

            QUESTION 1: question here
            Answer: short answer

            QUESTION 2: question here
            Answer: short answer



            These questions should be detailed and solely based on the information provided in the document.


            <Begin Document>
            {doc}
            <End Document>"""


    
    prompt = PromptTemplate(
    input_variables=["doc","num"], template=prompt_template
    )
    llm = LLMChain(llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True), prompt=prompt)
    questions = (llm.run(num=num_of_questions,doc=document))
    #print(questions)
    if type == FLASH_CARDS:
        cleaned_questions = parse_flash_cards(questions)
    else:
        cleaned_questions = parse_questions(questions)
    #print(cleaned_questions)
    return cleaned_questions
    
def parse_flash_cards(questions):
    # Split the text by "QUESTION"
    parts = questions.split('QUESTION')[1:]  # ignore the first split as it will be an empty string before the first "QUESTION"
    
    # Initialize an empty list to hold the question-answer pairs
    qa_pairs = []
    
    # Iterate over the parts and extract the question and answer
    for part in parts:
        # Use regex to extract the question number, question, and answer
        match = re.search(r'\d+: (.*?)\nAnswer: (.*?)$', part, re.DOTALL)
        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            qa_pairs.append({'question': question, 'answer': answer})
    
    return qa_pairs    
def generate_questions(quiz_type:str,pages,no_of_questions):
    questions = []
    for page in pages:
        document = page.page_content
        cleaned_quest = generate_quizz(document,quiz_type,no_of_questions)
        questions.extend(cleaned_quest)

    return questions
    
def generate_page_ranges(n):
    ranges = {}
    for i in range(1, n, 10):
        end_page = min(i + 9, n)
        ranges[f"Pages {i} - {end_page}"] = [i-1,end_page]
    return ranges

def get_pages(selected_page_ranges,page_ranges_2_index,total_pages):
    if len(selected_page_ranges) == page_ranges_2_index:
        return total_pages
    
    pages_to_query = []
    for page_range in selected_page_ranges:
        from_index, to_index = page_ranges_2_index[page_range]
        for i in range(from_index,to_index):
            pages_to_query.append(total_pages[i])
    return pages_to_query
        
def get_selected_page_ranges(page_ranges):
    
    st.write("What Pages do you want to test your knowledge in?")
    # Create a row for 'Select All' checkbox
    
    col1, col2 = st.columns([1, 3])  # Adjust the ratio if needed
    select_all = col1.checkbox('Select All')

    if select_all:
        st.info("We advise you to test your knowledge per Page Range so your questions gets generated faster 😢😰")
        st.info("Should you want to still test your self on multiple Page Ranges, Try to Limit them as much as possible to get your questions generated faster 😒😪")
        selected_page_ranges = page_ranges
    else:
        selected_page_ranges = []
      
        for i in range(0, len(page_ranges), 6):
            cols = st.columns(6)  # Create 6 columns
            # Iterate over the columns
            for col, page_range in zip(cols, page_ranges[i:i+6]):
                with col:
                    if st.checkbox(page_range):
                        selected_page_ranges.append(page_range)

    return selected_page_ranges


    
def quizz_generation(pdf_name, category_dict):
    st.header(f"🌟 Welcome to StuddyBuddy! Ready to Test Your Knowledge and Practice What You've Learnt on {pdf_name}? 🚀")
    
    pages = [Document(page_content = t["page_content"]) for t in category_dict[pdf_name]["pages"]]
    
    # print("PAEGS is ",pages)
    page_ranges_2_index= generate_page_ranges(len(pages))

    page_ranges = list(page_ranges_2_index.keys())
    selected_page_ranges = get_selected_page_ranges(page_ranges)
    st.write('Selected Page Range:', ', '.join(selected_page_ranges) if selected_page_ranges else 'None')
    selection = st.radio("What Format do you want?", [OBJECTIVE, TRUE_OR_FALSE, FLASH_CARDS],horizontal=True)
    user_input = None
    if selection == OBJECTIVE:
        user_input = st.number_input('How many Questions Do you want Generated per Page', min_value=2, value=OBJECTIVE_DEFAULT_VALUE, step=1, format='%d')
        if user_input > OBJECTIVE_DEFAULT_VALUE:
            st.error(f"Error: value should not be greater than {OBJECTIVE_DEFAULT_VALUE}.")
            return
        st.info(f"You'll be tested on a maximum of {user_input} questions per Page")
    if selection == TRUE_OR_FALSE:
        user_input = st.number_input('How many Questions Do you want Generated per Page', min_value=2, value=TRUE_OR_FALSE_DEFAULT_VALUE, step=1, format='%d')
        if user_input > TRUE_OR_FALSE_DEFAULT_VALUE:
            st.error(f"Error: value should not be greater than {TRUE_OR_FALSE_DEFAULT_VALUE}.")
            return
        st.info(f"You'll be tested on a maximum of {TRUE_OR_FALSE_DEFAULT_VALUE} questions per Page")
    if selection == FLASH_CARDS:
        ""
        user_input = 2 # for test
        
        #flash card saga starts here
    if st.button('generate quizz'):
        if len(selected_page_ranges) == 0:
            st.error("Please select Page Ranges you want to get tested on")
            return  
        pages_to_query = get_pages(selected_page_ranges,page_ranges_2_index,pages)
        if selection == FLASH_CARDS:
             st.info("Generating flash cards..... Please wait")
        else:
            st.info("Generating Quizz Questions")
        st.session_state['questions'] = generate_questions(selection,pages_to_query,user_input)
        st.session_state['selection'] = selection
        log_activity('quiz-generation-succeeded')
        st.success("Quizz generated Successfullyy")
        st.info("Please Click on the Display Quizz Button to Proceed")
        return "Done"
    
#if 'quiz_state' not in st.session_state:
#    st.session_state.quiz_state = {'submitted': False, 'user_answers': {}, 'score': 0}


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# callbacks
def callback():
    st.session_state.button_clicked = True


def callback2():
    st.session_state.button2_clicked = True


def get_state():
    return {'submitted': False, 'user_answers': {}, 'score': 0}

def display_flash_card():
    if 'questions' not in st.session_state:
        st.error("Generate Flash Cards please")
        return
    questions = st.session_state['questions']
    type = st.session_state['selection']

def display_on_streamlit():
    if 'questions' not in st.session_state:
        st.error("Generate Quizz please")
        return
    
    st.warning("For this demo, leaving this page means you have to generate the quizzes again")
    log_activity('start-quiz-viewing')
    data = st.session_state['questions']
    type = st.session_state['selection']
    if type == FLASH_CARDS:
        ""
        for i in range(len(data)):
            data[i]["num"] = i+1

        local_css('./style.css')
        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False

        if "button2_clicked" not in st.session_state:
            st.session_state.button2_clicked = False

        if "q_no" not in st.session_state:
            st.session_state.q_no = 0

        if "q_no_temp" not in st.session_state:
            st.session_state.q_no_temp = 0
        tab1 = st.tabs(["Flashcards"])
        question_nums = []
        with tab1[0]:
            noq = len(data)
            st.caption("There are {} flashcard questions for you to revise with".format(noq))
            col1, col2 = st.columns(2)
            with col1:
                question = st.button(
                    "Draw question", on_click=callback, key="Draw", use_container_width=True
                )
            with col2:
                answer = st.button(
                    "Show answer", on_click=callback2, key="Answer", use_container_width=True
                )
            if question or st.session_state.button_clicked:
                while True:
                    random_question = random.choice(data)
                    num = random_question['num']
                    if num not in question_nums:
                        question_nums.append(num)
                        break
                    if len(question_nums) == noq:
                        question_nums = []

                
                st.session_state.q_no = question_nums[-1]

                # this 'if' checks if algorithm should use value from temp or new value (temp assigment in else)
                if st.session_state.button2_clicked:
                    st.markdown(
                        f'<div class="blockquote-wrapper"><div class="blockquote"><h1><span style="color:#ffffff">{data[st.session_state.q_no_temp-1]["question"]}</span></h1><h4>&mdash; Question no. {st.session_state.q_no_temp}</em></h4></div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="blockquote-wrapper"><div class="blockquote"><h1><span style="color:#ffffff">{data[st.session_state.q_no-1]["question"]}</span></h1><h4>&mdash; Question no. {st.session_state.q_no}</em></h4></div></div>',
                        unsafe_allow_html=True,
                    )
                    # keep memory of question number in order to show answer
                    st.session_state.q_no_temp = st.session_state.q_no

                if answer:
                    st.markdown(
                        f"<div class='answer'><span style='font-weight: bold; color:#6d7284;'>Answer to question number {st.session_state.q_no_temp}</span><br><br>{data[st.session_state.q_no_temp-1]['answer']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.session_state.button2_clicked = False

            # this part normally should be on top however st.markdown always adds divs even it is rendering non visible parts?

            st.markdown(
                '<div><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Abril+Fatface&family=Barlow+Condensed&family=Cabin&display=swap" rel="stylesheet"></div>',
                unsafe_allow_html=True,
            )



    else:
        if type == OBJECTIVE:
            st.title("Multiple Choice Quiz")
        elif type == TRUE_OR_FALSE:
            st.title("True or False Type Quiz")

        state = get_state()

        if not state['submitted']:
            form_placeholder = st.empty()
            with form_placeholder.form(key='quiz_form'):
                user_answers = {}
                for idx, question in enumerate(data, start=1):
                    st.write(f"Question {idx}: {question['question']}")
                    options = question['options']
                    user_answers[idx] = st.radio(f"Options for Question {idx}", options.keys(), format_func=lambda x: options[x])
                
                submit_button = st.form_submit_button(label='Submit Answers')

            if submit_button:
                state['submitted'] = True
                state['user_answers'] = user_answers
                # Calculate score
                state['score'] = sum(1 for idx, q in enumerate(data, start=1) if user_answers[idx] == q['answer'])
                form_placeholder.empty() # Clear the form

        if state['submitted']:
            for idx, question in enumerate(data, start=1):
                user_answer = state['user_answers'][idx]
                correct_answer = question['options'][question['answer']]
                if user_answer == question['answer']:
                    feedback = "Correct"
                else:
                    feedback = f"Incorrect, Correct answer is {correct_answer}"

                st.write(f"Question {idx}: {question['question']}")
                st.write(f"Your answer: {question['options'][user_answer]} - {feedback}")
                st.write("\n")
                st.write("\n")

            # Display score
            st.write(f"Your Score: {state['score']}/{len(data)}")
            log_activity('Finished-quiz')

            if st.button('Try Again'):
                state['submitted']
                st.experimental_rerun()


def clear_prev_gen_quiz():
    if 'questions' in st.session_state:
        del st.session_state['questions']
        del st.session_state['selection']