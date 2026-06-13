import streamlit as st

st.title("Hola mundo")
st.write("Esta es mi primera aplicación desarrollada con Streamlit.")

nombre = st.text_input("Escriba su nombre:")

if nombre:
    st.write(f"¡Hola, {nombre}!")