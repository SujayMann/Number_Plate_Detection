import streamlit as st

def info_page():
    st.title("How to Use the App")
    st.write("""
        This app detects a license plate from an uploaded image.
        
        **Steps**:
        1. Navigate to the **Detect** page via the sidebar.
        2. Select between **Take a picture** and **Upload a picture**.
        3. If you want to **Take a picture**, then allow camera access when prompted upon clicking the **Enable Camera** checkbox.
        4. If you want to **Upload a picture**, use the **Browse files** button to do so.
        5. Upon capturing or uploading the picture, the model detects if a number plate is present in the picture or not.
        6. If there is one, the model gives the text of the number plate as output.
    """)

    st.write("""If the app doesn't work, it is most likely due to unavailability of the model used for prediction.
    The model file was uploaded using Git LFS free plan so if that exceeds the restrictions then the app might not work temporarily.
    Apologies for the inconvenience caused.""")

if __name__ == '__page__':
    info_page()
