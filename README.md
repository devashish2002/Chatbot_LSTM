# Chatbot_LSTM
A chatbot based on the Long Short Term Memory (LSTM) neural network. (Inspired from http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/)

Sequence To Sequence model introduced in *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation* has since then, become the Go-To model for Dialogue Systems and Machine Translation. It consists of two RNNs (Recurrent Neural Network) : An Encoder and a Decoder. The encoder takes a sequence(sentence) as input and processes one symbol (word) at each timestep. Its objective is to convert a sequence of symbols into a fixed size feature vector that encodes only the important information in the sequence while losing the unnecessary information. We can visualize data flow in the encoder along the time axis, as the flow of local information from one end of the sequence to another.

![screenshot from 2017-12-04 14-50-52](https://user-images.githubusercontent.com/24549241/33545047-f4c35c3a-d902-11e7-913f-7d8db097639e.png)

Each hidden state influences the next hidden state and the final hidden state can be seen as the summary of the sequence. This state is called the context or thought vector, as it represents the intention of the sequence. From the context, the decoder generates another sequence, one symbol (word) at a time. Here, at each time step, the decoder is influenced by the context and the previously generated symbols.

## Padding
Before training, we work on the dataset to convert the variable length sequences into fixed length sequences, by *padding*. We use a few special symbols to fill in the sequence.

1. EOS : End of sentence
2. PAD : Filler
3. GO : Start decoding
4. UNK : Unknown; word not in vocabulary

![screenshot from 2017-12-04 14-52-41](https://user-images.githubusercontent.com/24549241/33545052-fa539642-d902-11e7-9aba-f8c57e3c5eeb.png)

Assuming that we would like our sentences (queries and responses) to be of fixed length, 10, this pair will be converted to:

![screenshot from 2017-12-04 14-52-57](https://user-images.githubusercontent.com/24549241/33545058-fcae7ef2-d902-11e7-89c5-dc5a22ff079a.png)

## Word Embedding
Word Embedding is a technique for learning dense representation of words in a low dimensional vector space. Each word can be seen as a point in this space, represented by a fixed length vector. Semantic relations between words are captured by this technique.

screenshot from 2017-12-04 14-57-55](https://user-images.githubusercontent.com/24549241/33545269-92ced2d8-d903-11e7-9669-c7bea4798b37.png)

Word Embedding is typically done in the first layer of the network : Embedding layer, that maps a word (index to word in vocabulary) from vocabulary to a dense vector of given size.




