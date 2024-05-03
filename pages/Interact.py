import streamlit as st
import os
import re
import random
import traceback
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain.chains import RetrievalQA
#from langchain_openai import OpenAIEmbeddings
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
import pickle as pk
import numpy as np
from utils import log_activity
import pandas as pd
import google.generativeai as genai
import textwrap


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OBJECTIVE="Objective"
TRUE_OR_FALSE="True/False"
FLASH_CARDS='Flash Cards😍'
FILL_IN_GAP = "Fill in the gaps"
OBJECTIVE_DEFAULT_VALUE=5
TRUE_OR_FALSE_DEFAULT_VALUE=5

llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

#llm = ChatOpenAI(    
#        openai_api_key=OPENAI_API_KEY, 
#        model_name="gpt-4", 
#        temperature=0.0
#   )

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
    You are a helpful and informative bot that answers questions related to a pdf which you are aware has been uploaded. 
    1. If the question is a greeting like "Hi" or a generic sentence like "How are you" and reply accordingly. Therefore:
        Example: 
            Question: Hi
            PASSAGE OF PDF: Any passage
            ANSWER: Reply the greeting with a good response and then prompt user to ask about the pdf which is uploaded already. That is all you should do and add no other context.
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
                func=self.pdf_agent.invoke,
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

# Upload Section
def upload():
    #if 'loggedIn' not in st.session_state:
    #    st.write("Looks Like you are not logged in 🙁😕")
    #    st.error("Please Login to access this screen")
    #    return
    st.header('Upload Your PDF')
    st.write("Drag and drop your PDF file here or click to upload. Please ensure that the text in the PDF is selectable and not a scanned image.")
    uploaded_file = st.file_uploader("", type="pdf")

    if uploaded_file is not None:
        # Check if the uploaded file is a PDF
        if "pdf_name" in st.session_state:
            if 'past' in st.session_state:
                del st.session_state.past[:]
            if 'generated' in st.session_state:
                del st.session_state.generated[:]
        if 'questions' in st.session_state:
            del st.session_state['questions']


        pdf = uploaded_file.name
        pdf_name = '.'.join(pdf.split('.')[:-1])
        with open(pdf, mode='wb') as f:
            f.write(uploaded_file.getbuffer()) # save pdf to disk
        st.success("Uploading File.....")
        loader = PyPDFLoader(pdf)
        st.session_state['pages'] = loader.load_and_split()
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=20)
        texts = text_splitter.split_documents(documents)
        if len(texts) == 0:
            st.error("Please ensure your uploaded  document is selectable (i.e not scanned)")
        else:
            doc = [d.dict() for d in loader.load_and_split()]
            df = pd.DataFrame()
            df["Titles"] = [f"split{i+1}" for i in range(len(doc))]
            df["page_content"] = pd.DataFrame(doc)["page_content"]
            st.success("File uploaded successfully!")
            st.write("Processing Uploaded PDF..........please wait till success message")
            df['Embeddings'] = df.apply(lambda row: embed_fn(row['Titles'], row['page_content']), axis=1)
            st.session_state["curr_df"] = df
            st.session_state['pdf_name'] = '.'.join(pdf.split('.')[:-1])
            st.success("PDF processed Successfully!!!")
            st.info("Please proceed to bot")

def chatbot():
    if 'curr_df' not in st.session_state:
        #if 'loggedIn' not in st.session_state:
        #    st.write("Looks Like you are not logged in 🙁😕")
        #    st.error("Please Login to access this screen")
        #    return
        st.error("Please Upload a PDF File")
        
    else:
        df = st.session_state["curr_df"]
        st.header("Q & A Assistant")
        #qa = st.session_state['qabot']
        st.markdown(f"Ask questions related to the uploaded file here : {st.session_state['pdf_name']}")
        message("Hiiiiii!!, I am your studyBud, your personal reading buddy  😊 😊 😊!!!!")
        if "past" not in st.session_state:
            st.session_state['past'] = []
        if "generated" not in st.session_state:
            st.session_state["generated"] = []
        if "input_message_key" not in st.session_state:
            st.session_state["input_message_key"] = str(random.random())
        chat_container = st.container()
        user_input = st.text_input("Type your question here.", key=st.session_state["input_message_key"])
        if st.button("Send"):
            passage = find_best_passage( user_input, df)
            prompt = make_prompt(user_input, passage)
            model = genai.GenerativeModel('gemini-pro')
            if 'chat' in st.session_state:
                response = st.session_state["chat"].send_message(prompt)
            else:
                chat = model.start_chat(history=[])
                response = chat.send_message(prompt)
                st.session_state["chat"] = chat

            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(response.text)
            st.session_state["input_message_key"] = str(random.random())
            st.rerun()
        if st.session_state["generated"]:
             with chat_container:
                  for i in range(len(st.session_state["generated"])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                    message(st.session_state["generated"][i], key=str(i))


def parse_questions(input_text):
    question_blocks = input_text.split('QUESTION')

    questions = []

    for block in question_blocks:
        print("BLOCK is", block.replace("*", ""))
        if block.strip() and "CHOICE_A" in block:
            block = block.replace("*", "")
            question_num = int(re.search(r'\d+', block).group())

          
            pattern = r"\d:\s*(.*?)(?:\nCHOICE_|Answer:)"
            # Match the pattern in the text
            match = re.search(pattern, block, re.DOTALL)
            if match:
                question_text = match.group(1).replace("*", "").replace("\n", "")

         
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

def parse_german(input_text):
    question_blocks = input_text.replace("*", "").split('QUESTION')
    print(question_blocks, len(question_blocks))

    questions = []

    for block in question_blocks[1:]:
        print("BLOCK is", block.replace("*", ""))

        pattern = r'[0-9]'
        temp = ''.join([p for p in block if p not in ["*", "\n", ":"]]).split("Answer")
        print(temp)
        question_number, question , answer = re.match(pattern,temp[0].strip()).group(),re.sub(pattern, '', temp[0].strip()), temp[1].strip()
        print(question_number, question , answer)
        question_obj = {
            "question_num": question_number,
            "question": question,
            "answer": answer.split(",")
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
            The questions should also follow format given in example with no extra asterisks in the questions or answers
            Example of bad format which must not be included in questions:
                **QUESTION 2:**

                What factors primarily control the partitioning of metals between sediments and water columns in river systems?

                CHOICE_A: NONE
                CHOICE_B: Temperature and salinity
                CHOICE_C: Redox potential and pH
                CHOICE_D: Flow velocity and turbidity
                CHOICE_E: Organic matter content and particle size

                **Answer:** C
            The sample above must never be returned in any generated question

            <Begin Document>
            {doc}
            <End Document>"""
    elif type == TRUE_OR_FALSE:
        prompt_template =  """
        You are a teacher preparing True or False questions for a quiz. Given the following document, please generate {num} True or False type questions  with 3 options and a corresponding True or False answer except for CHOICE_A which should ALWAYS BE defaulted to NONE
            For each question, randomly select a section or sentence from the document, and then create a question that is directly based on the text. please do not make any reference of the word "document". Please do not include asterisks(**) anywhere.

            Remember to vary the type of information each question is based on. Also, ensure each question has a different structure or focus to avoid repetition. All questions must be questions where options is True or False

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

            <Begin Document>
            {doc}
            <End Document>
            
            These questions should be detailed and solely based on the information provided in the document.
            The questions should also follow format given in example with no extra asterisks in the questions or answers and all answers should be an option between B or C
            Sample of bad format which must not be included in questions:

                **QUESTION 2:** Question here

                CHOICE_A: NONE
                CHOICE_B: choice here
                CHOICE_C: choice here

                **Answer:** C
            The sample above where asterisks (**) are added to either question or answer or both must never be returned in any generated question no matter what"""
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
            Also the questions should be questions with an answer so there should be no question where the answer is "Not provided in the given document."


            <Begin Document>
            {doc}
            <End Document>"""
    elif type == FILL_IN_GAP:
        prompt_template =  """You are a teacher preparing german (fill in the gap) type questions for a quiz. Given the following document (document starts from <Begin Document>), please generate {num} questions and a corresponding short answer 
            For each question, randomly select a section or sentence from the document, and then remove a number of random words from this selection but removed words must not be more than 5, please do not make any reference of the word "document"
            Then the question is the sentence/selection without removed words where there is an evident blank space which students are expected to fill
            The answers are the removed words.

            Remember to vary the type of information each question is based on. Also, ensure each question has a different structure or focus to avoid repetition.

            The examples below are to direct you on how to set the questions but on no ocassion are they ever to be included in the questions you generate.
            Example:

            Example Random selected statement : The essensce of ABC is for children

            Example QUESTION 1: The ______ of ABC is for  ____ (removed words are essence and children)
            Example Answer: essence, children

            Example 2
            Example Random selected statement : The name of the current team lead is Usman Ade

            Example QUESTION 2: The name of the current team lead is ______ (removed word is Usman Ade)
            Example Answer: Usman Ade
            
            End of example. Again These Examples are NEVER to be shown in the actual generated questions.

            These questions should be detailed and solely based on the information provided in the document, and should follow format of examples above withe "QUESTION" being in uppercase
            , however, NEVER include the removed words alongside the questions.
            Also the questions should be questions with an answer so there should be no question where the answer is "Not provided in the given document. Therefore, if the length of the document is not up to 20 words and is not enough to generate good questions, then return 'Sorry, the provided document is too short to generate meaningful fill-in-the-gap questions.'
            "


            <Begin Document>
            {doc}
            <End Document>"""
    
    prompt = PromptTemplate(
    input_variables=["doc","num"], template=prompt_template
    )

    llm = LLMChain(llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True), prompt=prompt)
    
    questions = (llm.run(num=num_of_questions,doc=document))
    print("questions===", questions)
    if type == FLASH_CARDS:
        cleaned_questions = parse_flash_cards(questions)
    elif type == FILL_IN_GAP:
        cleaned_questions = parse_german(questions)
    else:
        cleaned_questions = parse_questions(questions)
    print(cleaned_questions)
    return cleaned_questions
    
def parse_flash_cards(questions):
    # Split the text by "QUESTION"
    parts = questions.split('QUESTION')[1:]  # ignore the first split as it will be an empty string before the first "QUESTION"
    #print(parts)
    
    # Initialize an empty list to hold the question-answer pairs
    qa_pairs = []
    
    # Iterate over the parts and extract the question and answer
    for part in parts:
        # Use regex to extract the question number, question, and answer
        pattern = r'[0-9]'
        temp = ''.join([p for p in part if p not in ["*", "\n", ":"]]).split("Answer")
        question , answer = re.sub(pattern, '', temp[0].strip()), temp[1].strip()
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


    
def quizz_generation():
    if 'curr_df' not in st.session_state:
        st.error("A PDF File has to be uploaded to generate quizz")
        return
    st.header("🌟 Welcome to StuddyBuddy! Ready to Test Your Knowledge and Practice What You've Learnt? 🚀")
    
    pages = st.session_state['pages']
    # print("PAEGS is ",pages)
    page_ranges_2_index= generate_page_ranges(len(pages))

    page_ranges = list(page_ranges_2_index.keys())
    selected_page_ranges = get_selected_page_ranges(page_ranges)
    st.write('Selected Page Range:', ', '.join(selected_page_ranges) if selected_page_ranges else 'None')
    selection = st.radio("What Format do you want?", [OBJECTIVE, TRUE_OR_FALSE, FLASH_CARDS,FILL_IN_GAP],horizontal=True)
    user_input = None
    if selection == OBJECTIVE:
        user_input = st.number_input('How many Questions Do you want Generated per Page', min_value=2, value=OBJECTIVE_DEFAULT_VALUE, step=1, format='%d')
        if user_input > OBJECTIVE_DEFAULT_VALUE:
            st.error(f"Error: value should not be greater than {OBJECTIVE_DEFAULT_VALUE}.")
            return
        st.info(f"You'll be tested on a maximum of {user_input} questions per Page")
    if selection == TRUE_OR_FALSE or selection == FILL_IN_GAP:
        user_input = st.number_input('How many Questions Do you want Generated per Page', min_value=2, value=TRUE_OR_FALSE_DEFAULT_VALUE, step=1, format='%d')
        if user_input > TRUE_OR_FALSE_DEFAULT_VALUE:
            st.error(f"Error: value should not be greater than {TRUE_OR_FALSE_DEFAULT_VALUE}.")
            return
        st.info(f"You'll be tested on a maximum of {user_input} questions per Page")
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
        print(st.session_state['questions'])
        st.session_state['selection'] = selection
        st.success("Quizz generated Successfullyy")
        st.info("Please Click on the Display Quizz Button to Proceed")
    
if 'quiz_state' not in st.session_state:
    st.session_state.quiz_state = {'submitted': False, 'user_answers': {}, 'score': 0}


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
    
    elif type == FILL_IN_GAP:
        state = get_state()
        if not state['submitted']:
            form_placeholder = st.empty()
            with form_placeholder.form(key='quiz_form'):
                st.write("Separate multiple answers with a comma. Please ensure your answers are exacly the same as the one in the pdf")
                user_answers = {}
                for idx, question in enumerate(data, start=1):
                    st.write(f"Question {idx}: {question['question']}")
                    #corr_ans = question['answer']
                    user_answers[idx] = st.text_input("Wirte your answer here separated by a comma", key=idx)
                
                submit_button = st.form_submit_button(label='Submit Answers')
                
            if submit_button:
                state['submitted'] = True
                state['user_answers'] = user_answers
                # Calculate score
                score = 0
                for i, dets in enumerate(data, start = 1):
                    if len(dets['answer']) != len(user_answers[i].split(",")):
                        continue
                    else:
                        for ac, sel in zip(dets['answer'], user_answers[i].split(",")):
                            if sel.strip().lower() == ac.strip().lower():
                                score += 1
                    
                state['score'] = score
                form_placeholder.empty() # Clear the form
                
        if state['submitted']:
            for idx, question in enumerate(data, start=1):
                user_answer = state['user_answers'][idx].split(",")
                correct_answer = question['answer']
                
                feedback = f"Correct answer is {correct_answer}"
                st.write(f"Question {idx}: {question['question']}")
                st.write(f"Your answer: {user_answer} - {feedback}")
                st.write("\n")
                st.write("\n")

            # Display score
            st.write(f"Your Score: {state['score']}/{len(data)}")

            if st.button('Try Again'):
                state['submitted']
                st.experimental_rerun()
                
        


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

            if st.button('Try Again'):
                state['submitted']
                st.experimental_rerun()


def main():
    #log_activity("visit-interact-page")
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Upload File","Interact With Uploaded PDF File", "Quizz Generation","Display Quizz"])
    if selection == "Upload File":
        #log_activity("visit-interact-page")
        upload()
    elif selection == "Interact With Uploaded PDF File":
        #log_activity('select-interaction-chat')
        chatbot()
    elif selection == "Quizz Generation":
        #log_activity('quiz-generation')
        quizz_generation()
    elif selection == "Display Quizz":
        #log_activity('display-quiz')
        display_on_streamlit()

main()