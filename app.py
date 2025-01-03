import streamlit as st

def main():
    home = st.Page('info.py', title='Home', icon='🏠')
    predict = st.Page('prediction.py', title='Detect', icon='🎫')
    pages = st.navigation([home, predict])
    pages.run()
    
if __name__ == '__main__':
    main()
