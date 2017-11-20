import re
import pandas as pd
from lxml import etree


def get_data(file_name,genre):
    sentences = []
    with (open('./data/ar_PAN/{}.xml'.format(file_name),'r',encoding='utf8')) as f:
        doc = etree.parse(f)
    root = doc.getroot()
    for i in range(len(root[0])):
        sentences.append([preprocessing(root[0][i].text),genre])
    return sentences

def get_tweets(df):
    sentences = []
    for index, row in df.iterrows():   
        sentences +=get_data(row.id,row.genre)
    return pd.DataFrame(sentences,columns=['text','label'])
    
def load_data():
    train = pd.read_csv('./data/train_data.csv',encoding='utf8')
    train = train[['id','genre']].copy()
    test= pd.read_csv('./data/test_data.csv',encoding='utf8')
    test = test[['id','genre']].copy()
    df_train = get_tweets(train)
    df_test = get_tweets(test)
    x_train, y_train = df_train[['text']].values,df_train[['label']].values
    x_test, y_test = df_test[['text']].values,df_test[['label']].values
    return x_train, y_train, x_test, y_test

def remove_diacritics(string):
    regex = re.compile(r'[\u064B\u064C\u064D\
    \u064E\u064F\u0650\u0651\u0652]')
    return re.sub(regex, ' ', string)
def remove_urls(string):
    regex = re.compile(r"(http|https|ftp)://\
    (?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|\
    (?:%[0-9a-fA-F][0-9a-fA-F]))+")
    return re.sub(regex, ' ', string)
def remove_numbers(string):
    regex = re.compile(r"(\d|[\u0660\u0661\u0662\
    \u0663\u0664\u0665\u0666\u0667\u0668\u0669])+")
    return re.sub(regex, ' ', string)
def noramlize(string):
    regex = re.compile(r'[إأٱآا]')
    string = re.sub(regex, 'ا', string)
    regex = re.compile(r'[ى]')
    string = re.sub(regex, 'ي', string)
    regex = re.compile(r'[ؤئ]')
    string = re.sub(regex, 'ء', string)
    return string
def remove_non_arabic_words(string):
    return ' '.join([word for word in \
            string.split() if not re.findall(
        r'[^\s\u0621\u0622\u0623\u0624\u0625\
        \u0626\u0627\u0628\u0629\u062A\u062B\
        \u062C\u062D\u062E\u062F\u0630\u0631\
        \u0632\u0633\u0634\u0635\u0636\u0637\
        \u0638\u0639\u063A\u0640\u0641\u0642\
        \u0643\u0644\u0645\u0646\u0647\u0648\
        \u0649\u064A]',
        word)])
def remove_extra_whitespace(string):
    string = re.sub(r'\s+', ' ', string)
    return re.sub(r"\s{2,}", " ", string).strip()

def remove_non_arabic_symbols(string):
    return re.sub(r'[^\u0600-\u06FF]', '', string)

def remove_dubplicated_letters(string):
    return re.sub(r'(.)\1{2,}', r'\1', string)

def preprocessing(string):
    # remove diacritics
    string = remove_diacritics(string)
    # remove non arabic symbols
    #string = remove_non_arabic_symbols(string)
    # remove punctiations
    #string = remove_punctiation(string)
    # remove dubplicated letters
    string = remove_dubplicated_letters(string)
    # remove non arabic words
    string = remove_non_arabic_words(string)
    # normelize the text
    string = noramlize(string)
    # remove extra white spaces
    string = remove_extra_whitespace(string)
    return string

def prepare_tokenized_data(data, max_num_words, max_sequence_length):
    train, test = data
    if not os.path.exists('data/tokenizer.pkl'):
        tokenizer = Tokenizer(num_words=max_num_words)
        tokenizer.fit_on_texts(train)

        with open('data/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

        print('Saved tokenizer.pkl')
    else:
        with open('data/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            print('Loaded tokenizer.pkl')

    sequences_train = tokenizer.texts_to_sequences(train)
    sequences_test = tokenizer.texts_to_sequences(test)
    word_index = tokenizer.word_index
    print('Found %s unique 1-gram tokens.' % len(word_index))
    print('Min sequence length: {}'.format(np.min(list(map(len, sequences_train)))))
    print('Average sequence length: {}'.format(np.mean(list(map(len, sequences_train)), dtype=int)))
    print('Max sequence length: {}'.format(np.max(list(map(len, sequences_train)))))
    train = pad_sequences(sequences_train, maxlen=max_sequence_length)
    test = pad_sequences(sequences_test, maxlen=max_sequence_length)
    return (train, test, word_index)

def load_embedding_matrix(embedding_path, word_index, embedding_dim,gensim=True):
    if not os.path.exists('data/embedding_matrix.npy'):
        print('Load embedding model.')
        if gensim:
            word2vec = KeyedVectors.load(embedding_path)
        else:
            
            word2vec = KeyedVectors.load_word2vec_format(embedding_path)
        embeddings_index = {}
        for i in range(len(word2vec.index2word)):
            word = word2vec.index2word[i]
            vector = word2vec.index2word[i]
            embeddings_index[word] = word2vec[vector]
        print('Numbers of words in embedding model :', len(embeddings_index))
        print('Preparing embedding matrix.')
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        print('Embeding matrix size :', len(embedding_matrix))
        np.save('data/embedding_matrix.npy', embedding_matrix)
        print('Saved embedding matrix')
    else:
        embedding_matrix = np.load('data/embedding_matrix.npy')
        print('Loaded embedding matrix')
    return embedding_matrix