#!/usr/bin/env python3
"""
Streamlit Web App for AI Chatbot
A conversational AI interface using transformer models
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
import random

# Configure page
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🤖 AI Conversational Chatbot")
st.markdown("""
Powered by state-of-the-art transformer models for natural conversation.
Chat with our AI assistant about anything you'd like to know!
""")

@st.cache_resource
def load_chatbot():
    """Load the conversational model with caching"""
    try:
        # Use DialoGPT model for conversation
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_response(tokenizer, model, chat_history, user_input, max_length=1000):
    """Generate chatbot response"""
    try:
        # Encode the conversation history
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        
        # Append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history, new_user_input_ids], dim=-1) if chat_history is not None else new_user_input_ids
        
        # Generate response
        with torch.no_grad():
            chat_history_ids = model.generate(
                bot_input_ids, 
                max_length=max_length,
                num_beams=3,
                no_repeat_ngram_size=2,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        return response, chat_history_ids
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}", None

def get_sample_conversations():
    """Get sample conversation starters"""
    return {
        "General Chat": "Hello! How are you doing today?",
        "Technology": "What do you think about the future of artificial intelligence?",
        "Science": "Can you explain quantum computing in simple terms?",
        "Philosophy": "What do you think is the meaning of life?",
        "Creative Writing": "Can you help me write a short story about space exploration?",
        "Problem Solving": "I'm facing a difficult decision at work. Can you help me think through it?"
    }

def main():
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model_chat_history' not in st.session_state:
        st.session_state.model_chat_history = None
    if 'conversation_count' not in st.session_state:
        st.session_state.conversation_count = 0

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Chat Settings")
        
        # Model settings
        max_length = st.slider(
            "Max Response Length", 
            min_value=100, 
            max_value=2000, 
            value=1000, 
            help="Maximum length of the generated response"
        )
        
        # Conversation management
        st.markdown("---")
        st.subheader("💬 Conversation")
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.model_chat_history = None
            st.session_state.conversation_count = 0
            st.rerun()
        
        # Display conversation stats
        st.metric("Messages Exchanged", len(st.session_state.chat_history))
        
        st.markdown("---")
        st.markdown("""
        ### About
        This chatbot uses Microsoft's DialoGPT model to 
        generate human-like responses in conversation.
        
        **Features:**
        - 🎨 Natural conversation flow
        - 🧠 Context-aware responses  
        - 💬 Multi-turn conversations
        - ⚡ Fast response generation
        """)
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Chat history display
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.chat_history:
                for i, (role, message) in enumerate(st.session_state.chat_history):
                    if role == "user":
                        st.markdown(f"👤 **You:** {message}")
                    else:
                        st.markdown(f"🤖 **AI:** {message}")
                    
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("---")
            else:
                st.info("👋 Welcome! Start a conversation by typing a message below.")
        
        # Chat input
        user_input = st.text_input(
            "Your message:", 
            placeholder="Type your message here...",
            key="user_input"
        )
        
        col_send, col_clear = st.columns([1, 1])
        with col_send:
            send_button = st.button("💬 Send Message", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear Input", use_container_width=True):
                st.session_state.user_input = ""
                st.rerun()
    
    with col2:
        st.subheader("💡 Quick Starters")
        
        sample_conversations = get_sample_conversations()
        
        st.markdown("**Try these conversation starters:**")
        
        for topic, starter in sample_conversations.items():
            if st.button(f"{topic}", key=f"starter_{topic}", use_container_width=True):
                st.session_state.selected_starter = starter
                st.rerun()
        
        # Handle selected starter
        if hasattr(st.session_state, 'selected_starter'):
            user_input = st.session_state.selected_starter
            delattr(st.session_state, 'selected_starter')
    
    # Process user input
    if (send_button or hasattr(st.session_state, 'selected_starter')) and user_input:
        with st.spinner("🤖 AI is thinking..."):
            # Load model
            tokenizer, model = load_chatbot()
            
            if tokenizer and model:
                # Add user message to chat history
                st.session_state.chat_history.append(("user", user_input))
                
                # Generate AI response
                ai_response, new_chat_history = generate_response(
                    tokenizer, model, st.session_state.model_chat_history, user_input, max_length
                )
                
                # Update chat history
                st.session_state.chat_history.append(("ai", ai_response))
                st.session_state.model_chat_history = new_chat_history
                st.session_state.conversation_count += 1
                
                # Clear input and refresh
                st.rerun()
            else:
                st.error("Failed to load the chatbot model. Please refresh the page and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>Built with ❤️ using Streamlit and Transformers | 
    <a href="https://github.com/chandrikachandra30/ai-chatbot" target="_blank">View Source Code</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
