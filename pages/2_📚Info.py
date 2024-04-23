import streamlit as st

# Page title
st.title("Information about LSTM")

# Description of LSTM
st.image("pages/lstm.png", use_column_width=True)

st.markdown("""
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) architecture, designed to overcome the limitations of traditional RNNs in capturing long-term dependencies in sequential data.

### How LSTM Works:
LSTM networks have a more complex architecture compared to traditional RNNs. They incorporate memory cells and gating mechanisms to control the flow of information through the network. The key components of an LSTM unit include:

1. **Cell State (Ct):** The cell state serves as the memory of the network and runs straight down the entire chain. It can selectively add or remove information using gates.
2. **Forget Gate (ft):** Decides what information to discard from the cell state.
3. **Input Gate (it):** Determines which values to update.
4. **Output Gate (ot):** Controls what gets outputted based on the cell state.

LSTM networks are trained using backpropagation through time (BPTT) and gradient descent. They are well-suited for a wide range of sequential data tasks, including time series prediction, natural language processing, and speech recognition.

### Advantages of LSTM:
- Ability to capture long-term dependencies.
- Can handle vanishing and exploding gradient problems better than traditional RNNs.
- Suitable for processing sequences of varying lengths.

### Applications of LSTM:
- Time series prediction (e.g., stock price forecasting).
- Natural language processing tasks (e.g., language translation, sentiment analysis).
- Speech recognition and generation.
- Handwriting recognition.
- Music composition.

LSTM networks have become a fundamental building block in deep learning models for sequential data analysis due to their effectiveness in capturing temporal patterns and dependencies.
""")

# References
st.subheader("References:")
st.markdown("""
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah
- [Long Short-Term Memory](https://en.wikipedia.org/wiki/Long_short-term_memory) on Wikipedia
- [Hochreiter, S.; Schmidhuber, J. (1997). "Long short-term memory"](https://dl.acm.org/doi/10.1162/neco.1997.9.8.1735) - Original LSTM paper by Hochreiter & Schmidhuber
""")

