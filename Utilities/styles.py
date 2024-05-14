import streamlit as st

class Styles:
    # Define custom styles for the title
    title_styles = {
        "card": {
                "width": "400px",
                "height": "400px",
                "border-radius": "60px",
                "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
            }
        }
    #margin: 10px;
    display_card_style = """
                    border: 1px solid #ccc;
                    border-radius: 10px;
                    margin-bottom: 10px;
                    padding: 10px;
                    box-shadow: 5px 5px 5px #888888;
                """
    header_style = """
        font-family: "Raleway", sans-serif;
        font-weight: 300;
        font-size: 40px;
        color: #080808;
    """


    def skip_two_lines(self):
        st.write('\n')
        st.write('\n')