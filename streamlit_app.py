import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def main():
    """
    Main entry point of the application.
    """

    # Initialize API key and model
    groq_api_key = os.environ.get('GROQ_API_KEY')
    model = 'llama3-8b-8192'

    # Initialize Groq client
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    # Set up Streamlit interface
    st.set_page_config(page_title="MedIA - Diagnóstico Inteligente", layout="wide")
    
    st.title("MedIA")
    st.write(
        "Seja bem-vindo ao MedIA! 🌟\n\n"
        "Sou um sistema de inteligência artificial treinado para auxiliar na análise de sintomas e direcionar você para o caminho certo. "
        "Baseado em suas respostas, tentarei traçar um panorama do que pode estar acontecendo."
    )

    # Initialize system prompt
    system_prompt = (
        "Você é um especialista em diagnósticos médicos. Baseado nos sintomas apresentados pelo usuário, "
        "personalize um possível diagnóstico. Se ele falar apenas um sintoma, não faça mais perguntas, "
        "mas coloque todas as doenças relacionadas a ela possíveis. Peça apenas informações relevantes."
    )

    # Initialize conversation memory
    conversational_memory_length = 50000
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=conversational_memory_length, memory_key="chat_history", return_messages=True
        )

    if 'history' not in st.session_state:
        st.session_state.history = []

    # Create layout for chat history and user input
    col1, col2 = st.columns([2, 1])

    with col1:
        # Display chat history
        if st.session_state.history:
            st.write("### Histórico da Conversa")
            for entry in st.session_state.history:
                st.markdown(entry)

    with col2:
        # User input
        st.subheader("Digite seus sintomas")
        user_input = st.text_area(
            "Apresente TODOS seus sintomas DETALHADAMENTE, a intensidade e quando iniciaram. (Ex. Dor de cabeça forte, febre, de 40 graus, irritação na garganta, vomito, etc.)",
            height=200,
            key='user_input'
        )

        # Submit button to handle Enter key simulation
        submit_button = st.button("Enviar", key='submit_button')

        # Check if the submit button was pressed or Enter key was simulated
        if submit_button or st.session_state.get('submit_on_enter', False):
            if user_input:
                # Append user input to history
                st.session_state.history.append(f"**Você**: {user_input}")

                # Create prompt template
                prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(content=system_prompt),
                        MessagesPlaceholder(variable_name="chat_history"),
                        HumanMessagePromptTemplate.from_template("{human_input}"),
                    ]
                )

                # Create conversation chain
                conversation = LLMChain(
                    llm=groq_chat,
                    prompt=prompt,
                    verbose=False,
                    memory=st.session_state.memory,
                )

                # Get response from the model
                response = conversation.predict(human_input=user_input)

                # Append response to history
                st.session_state.history.append(f"**MedIA**: {response}")

                # Clear user input after submission
                st.session_state['submit_on_enter'] = False
                st.experimental_rerun()
            else:
                st.warning("Por favor, insira seus sintomas antes de enviar.")

        # JavaScript snippet to simulate pressing the 'submit' button when Enter is pressed
        st.markdown("""
            <script>
            document.addEventListener('DOMContentLoaded', function() {
                const textarea = document.querySelector('textarea');
                textarea.addEventListener('keydown', function(event) {
                    if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault();
                        document.querySelector('button').click();
                    }
                });
            });
            </script>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
