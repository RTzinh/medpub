import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

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
    st.set_page_config(page_title="MedIA", layout="wide")
    
    st.title("MedIA")
    st.write(
        """
        Seja bem-vindo ao MedIA! \n\n
        Sou um sistema de inteligência artificial treinado para auxiliar na análise de sintomas e direcionar você para o caminho certo. 
        Baseado em suas respostas, tentarei traçar um panorama do que pode estar acontecendo.
        """
    )

    # Initialize system prompt
    system_prompt = (
        "Você A um especialista em diagnósticos médicos. Baseado nos sintomas apresentados pelo usuário, "
        "personalize um possível diagnóstico. Sugira ao paciente que ele responda todas as perguntas sem excessão, não dê a resposta enquanto ele não responder as perguntas. Após ele responder as 10 perguntas, pode dar o diagnostico"
        "coloque todas as doenças relacionadas a ela possíveis. Peça apenas informações relevantes, não faça perguntas muito especificas. faça sempre 10 perguntas muito uteis, nem menos nem mais que isso. Faça 1 pergunta de cada vez"
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
    col1, col2 = st.columns([1, 2])

    with col1:
        # User input area
        st.subheader("Digite seus sintomas")
        user_input = st.text_area(
            "Se possivel, apresente TODOS seus sintomas DETALHADAMENTE, a intensidade e quando iniciaram, para um diagnóstico mais preciso e rápido.",
            height=200,
            key='user_input'
        )

        # Submit button to handle Enter key simulation
        submit_button = st.button("Enviar", key='submit_button')

        if submit_button or st.session_state.get('submit_on_enter', False):
            if user_input:
                # Append user input to history
                st.session_state.history.append(f"<strong>Você:</strong> {user_input}")

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
                st.session_state.history.append(f"<strong>MedIA:</strong> {response}")

                # Clear user input after submission
                st.session_state['submit_on_enter'] = False
                st.experimental_rerun()
            else:
                st.warning("Por favor, insira seus sintomas antes de enviar.")

    with col2:
        # Display chat history with scroll and separators
        st.subheader("Respostas do MedIA")
        if st.session_state.history:
            st.markdown(
                f"""
                <div style="height: 400px; overflow-y: scroll;">
                    {"<hr>".join(st.session_state.history)}
                </div>
                """, 
                unsafe_allow_html=True
            )

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
