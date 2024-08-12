import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure the Google Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")

genai.configure(api_key=api_key)

# Initialize the Gemini models
text_model = genai.GenerativeModel("gemini-pro")

# Function to get a response from Gemini
def get_gemini_response(input_text):
    try:
        if input_text:
            # When only text is provided
            response = text_model.generate_content(input_text)
        else:
            return None
        return response.text
    except Exception as e:
        st.error(f"Error communicating with the Google Gemini API: {e}")
        return None

# Initialize Streamlit app
st.set_page_config(page_title="Drug Repurposing Using Knowledge Graphs",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;600;700&display=swap');

    body {
        font-family: 'SF Pro Display', sans-serif;
        background-color: #f0f5f0;
    }

    .header {
        background-color: #e8f5e9;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }

    .header h1 {
        font-size: 36px;
        color: #2e7d32;
        margin: 0;
    }

    .sidebar .sidebar-content {
        padding: 20px;
    }

    .sidebar .sidebar-content h2 {
        font-size: 24px;
        color: #2e7d32;
    }

    .main-content {
        padding: 20px;
    }

    .main-content h2 {
        font-size: 28px;
        color: #2e7d32;
        text-align: center;
    }

    .chatbot-response {
        color: white !important;
        background-color: #4caf50;
        padding: 10px;
        border-radius: 10px;
        margin-top: 20px;
    }

    .smooth-transition {
        transition: all 0.3s ease;
    }

    .button-primary {
        background-color: #2e7d32;
        color: #fff;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }

    .button-primary:hover {
        background-color: #1b5e20;
    }

    .download-button {
        background-color: #2e7d32;
        color: #fff;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
        margin-top: 20px;
    }

    .download-button:hover {
        background-color: #1b5e20;
    }
    </style>
""", unsafe_allow_html=True)

def p_title(title):
    st.markdown(f'<h2 class="smooth-transition">{title}</h2>', unsafe_allow_html=True)

# Header
st.markdown("""
    <div class='header'>
        <h1>Drug Repurposing Using Knowledge Graphs</h1>
    </div>
""", unsafe_allow_html=True)
# Sidebar for navigation and chatbot
with st.sidebar:
    st.header('I want to:')
    nav = st.radio('', ['Get Drugs Recommendations', 'Explore Knowledge Graph', 'Model Performance Metrics', 'Help'])

    st.text('')
    st.text('')

    st.header("Chat with Us!")
    input_text = st.text_input("Enter your question here:", placeholder="Type your question...", key="chat_input")
    if st.button("Send"):
        if input_text:
            response = get_gemini_response(input_text)
            if response:
                st.markdown(f'<div class="chatbot-response">{response}</div>', unsafe_allow_html=True)
                # Add to chat history
                if 'chat_history' not in st.session_state:
                    st.session_state['chat_history'] = []
                st.session_state['chat_history'].append(("You", input_text))
                st.session_state['chat_history'].append(("Bot", response))
            else:
                st.error("Failed to get a response from the chatbot")
        else:
            st.error("Please enter a question.")

# Navigation for different pages
if nav == 'Get Drugs Recommendations':
    st.text('')
    p_title('Get Drug Recommendations')
    st.text('')

    disease_selection = st.selectbox("Select a disease", ["Dengue", "Chagas", "Malaria", "Yellow Fever", "Leishmaniasis", "Filariasis", "Schistosomiasis"], 
                                     help='Select a disease and perform predictions of new drugs on it. This project is focused on seven vector-borne diseases (dengue, chagas, malaria, yellow fever, leishmaniasis, filariasis, and schistosomiasis), but it can be extended to additional ones.')
    model_selection = st.selectbox("Select an embedding model", ["TransE", "TransR", "TransH", "UM", "DistMult", "RESCAL", "ERMLP"],
                                  help='Knowledge graph embedding models can encode biological information in a single mathematical space. Select an embedding model and perform drug predictions on the target diseases. This project is focused on seven embedding models (TransE, TransR, TransH, UM, DistMult, RESCAL, and ERMLP), which can be extended to additional ones.')

    if st.button("Get Recommendations"):
        # Drug recommendations
        final_selection = disease_selection + model_selection
        ranking_file = pd.read_csv(f'embedding_models/{final_selection}.csv', sep=',')
        
        st.markdown("<h3 style='text-align: center; color:#F63366; font-size:28px;'><b>Drug Recommendations<b></h3>", unsafe_allow_html=True)
        st.write(ranking_file)
        st.markdown("<h3 style='text-align: center; color:#FFFFFF; font-size:18px;'><b>Model dependence affects how the values in the 'score' column should be read; typically, they cannot be taken directly as a probability.<b></h3>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color:#FFFFFF; font-size:18px;'><b>If the compound is listed as 'yes' at ClinicalTrials.gov for the target disorders, then it is listed as 'no' in the 'in_clinical_trials' column.<b></h3>", unsafe_allow_html=True)

        # Download recommendation file
        csv = ranking_file.to_csv(index=False)
        st.download_button(
            label="Download Recommendations",
            data=csv,
            file_name=f"{final_selection}_recommendations.csv",
            mime="text/csv"
        )

        # Disease direct connections in DRKG
        st.markdown("<h3 style='text-align: center; color:#F63366; font-size:28px;'><b>Direct Connections in DRKG for the Target Disease<b></h3>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color:#FFFFFF; font-size:18px;'><b>All of the GNBR subgraphs entities that are directly related to the target disease are listed below. Red-highlighted compounds are indicated.<b></h3>", unsafe_allow_html=True)

        # Visualize
        with st.container():
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            HtmlFile = open(f'graphs/knowledge_graph_{disease_selection}.html', 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            components.html(source_code, height=625, width=750)
            st.markdown("</div>", unsafe_allow_html=True)

        # Download graph HTML file
        st.download_button(
            label="Download Knowledge Graph",
            data=HtmlFile,
            file_name=f"{disease_selection}_knowledge_graph.html",
            mime="text/html"
        )

        # Model performance
        performance_file = pd.read_csv('embedding_models/performance_metrics.csv', sep=';')
        filtered_data = performance_file[performance_file['final_selection'] == final_selection]
        st.markdown("<h3 style='text-align: center; color:#F63366; font-size:28px;'><b>Embedding Model Performance<b></h3>", unsafe_allow_html=True)
        st.dataframe(filtered_data[['Measure', 'Value']], hide_index=True)

        # Plot performance metrics
        fig, ax = plt.subplots()
        filtered_data.plot(kind='bar', x='Measure', y='Value', ax=ax, color='skyblue', legend=False)
        plt.title('Performance Metrics')
        plt.xlabel('Measure')
        plt.ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

if nav == 'Explore Knowledge Graph':
    st.text('')
    p_title('Explore Knowledge Graph')
    st.text('')

    disease_selection = st.selectbox("Select a disease for Knowledge Graph Exploration", ["Dengue", "Chagas", "Malaria", "Yellow Fever", "Leishmaniasis", "Filariasis", "Schistosomiasis"],
                                     help='Select a disease to explore its knowledge graph. This feature allows you to visualize and analyze the knowledge graph of various diseases.')

    # Display the knowledge graph
    with st.container():
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        HtmlFile = open(f'graphs/knowledge_graph_{disease_selection}.html', 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height=625, width=750)
        st.markdown("</div>", unsafe_allow_html=True)

    # Download graph HTML file
    st.download_button(
        label="Download Knowledge Graph",
        data=HtmlFile,
        file_name=f"{disease_selection}_knowledge_graph.html",
        mime="text/html"
    )
if nav == 'Model Performance Metrics':
    st.text('')
    st.text('')

    performance_file = pd.read_csv('embedding_models/performance_metrics.csv', sep=';')
    st.markdown("<h3 style='text-align: center; color:#F63366; font-size:28px;'><b>Model Performance Metrics<b></h3>", unsafe_allow_html=True)
    st.dataframe(performance_file, hide_index=True)

    # Plot performance metrics
    fig, ax = plt.subplots()
    performance_file.plot(kind='bar', x='final_selection', y='Value', ax=ax, color='lightgreen', legend=False)
    plt.title('Performance Metrics of Embedding Models')
    plt.xlabel('Embedding Model')
    plt.ylabel('Value')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

if nav == 'Help':   
    st.text('')

    st.text('')

    st.markdown("""
        <h3 style="text-align: center; color:#F63366; font-size:24px;">Help & Documentation</h3>
        <p style="text-align: center;">Welcome to the Drug Repurposing Using Knowledge Graphs application. This tool helps you explore drug recommendations, knowledge graphs, and model performance metrics.</p>
        <ul>
            <li><b>Get Drugs Recommendations:</b> Select a disease and an embedding model to get drug recommendations and view the corresponding knowledge graph and performance metrics.</li>
            <li><b>Explore Knowledge Graph:</b> Visualize and analyze the knowledge graph for the selected disease.</li>
            <li><b>Model Performance Metrics:</b> View the performance metrics for various embedding models used in drug repurposing.</li>
            <li><b>Help:</b> Find documentation and guidance on using the application.</li>
        </ul>
    """, unsafe_allow_html=True)
