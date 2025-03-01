import streamlit as st
import Get_Model as GT
def main():
    st.set_page_config(layout="wide")
    st.title("Pseudocode to Code Converter")
    if 'pseudocode' not in st.session_state:
        st.session_state.pseudocode = ""
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Code")
        code = st.text_area("Enter your code:", height=400)
        code = GT.clean_spaces(code)
    with col2:
        st.subheader("Pseudocode")
        pseudocode = st.text_area("Generated pseudocode:", 
                                value=st.session_state.pseudocode, 
                                height=400)
    if st.button("Generate"):
        st.session_state.pseudocode = code
        st.session_state.pseudocode = GT.format_lines(parameter2=code)
if __name__ == "__main__":
    main()