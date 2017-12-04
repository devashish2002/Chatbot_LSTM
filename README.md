# Chatbot_LSTM
A chatbot based on the Long Short Term Memory (LSTM) neural network. (Inspired from http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/)

Sequence To Sequence model introduced in *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation* has since then, become the Go-To model for Dialogue Systems and Machine Translation. It consists of two RNNs (Recurrent Neural Network) : An Encoder and a Decoder. The encoder takes a sequence(sentence) as input and processes one symbol (word) at each timestep. Its objective is to convert a sequence of symbols into a fixed size feature vector that encodes only the important information in the sequence while losing the unnecessary information. We can visualize data flow in the encoder along the time axis, as the flow of local information from one end of the sequence to another.

Each hidden state influences the next hidden state and the final hidden state can be seen as the summary of the sequence. This state is called the context or thought vector, as it represents the intention of the sequence. From the context, the decoder generates another sequence, one symbol (word) at a time. Here, at each time step, the decoder is influenced by the context and the previously generated symbols.



