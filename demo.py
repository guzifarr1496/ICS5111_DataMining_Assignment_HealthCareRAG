import streamlit as st
from RAG_Module import query_rag  # Import the RAG function from the module

st.title("ICS5111 Healthcare RAG Demo")

user_query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if user_query.strip():
        response = query_rag(user_query)  # Call the RAG model function
        st.write("### Answer:")
        st.write(response)
    else:
        st.warning("Please enter a query.")
