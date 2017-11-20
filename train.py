import keras
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Bidirectional, GRU
from keras.models import Sequential
from TweetSentimentAnalysis.utils import *
from keras.metrics import binary_accuracy
import tensorflow as tf
from keras.callbacks import *
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.models import Model
embedding_dim = 300
max_num_words = 100000
max_sequence_length = 30
validation_split = 0.3
embedding_path = './data/embeddings/wiki.ar.vec'
full_dataset = True

x_train, y_train, x_test, y_test, word_index = prepare_data(max_num_words,
                                                            max_sequence_length,
                                                            validation_split,
                                                            full_dataset=full_dataset)
embedding_matrix = load_embedding_matrix(embedding_path, word_index, embedding_dim)
# print(tokenize_tweets(tweets, max_sequence_length))

with tf.name_scope('InputLayer'):
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
with tf.name_scope('EmbeddingLayer'):
    embedded_sequences =Embedding(len(word_index) + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=False)(sequence_input)

with tf.name_scope('BidirectionalGRULayer'):
    x = Bidirectional(GRU(150))(embedded_sequences)
    x = Dropout(0.5)(x)
with tf.name_scope('DenseLayer'):
    x = Dense(32, activation='relu')(x)

with tf.name_scope('OutputLayer'):
    preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[binary_accuracy])

tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)
history = model.fit(x_train,y_train,batch_size=128,epochs=5,validation_data=(x_test,y_test),verbose=2,callbacks=[tbCallBack])
