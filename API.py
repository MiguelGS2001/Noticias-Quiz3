import pandas as pd
import numpy as np
import streamlit as st
import pickle
import language_tool_python
import gensim
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from transformers import BertTokenizer, BertModel
import fasttext

@st.cache_resource
def carga_2():
    tool = language_tool_python.LanguageTool("en")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nlp = spacy.load("en_core_web_lg")
    stop_nltk = stopwords.words('english')
    stop_spacy = nlp.Defaults.stop_words
    stop_todas = list(stop_spacy.union(set(stop_nltk)))
    dictionary = gensim.corpora.Dictionary.load("dictionary.dict")
    stop_words = set(STOP_WORDS)
    return tool, nlp, stop_todas, dictionary, stop_words

@st.cache_resource
def carga():
    with open("lda_model.pkl", "rb") as file:
        lda_model = pickle.load(file)

    with open("lda_tfidf_model.pkl", "rb") as file:
        lda_tfidf_model = pickle.load(file)

    with open("modelo_kmeans_bert.pkl", "rb") as file:
        kmeans_new = pickle.load(file)

    with open("modelo_kmeans.pkl", "rb") as file:
        km_new = pickle.load(file)

    with open("pca_fast.pkl", "rb") as file:
        pca = pickle.load(file)

    with open("scaler_fast.pkl", "rb") as file:
        scaler = pickle.load(file)
    
    ft_model = fasttext.load_model("fasttext_model.bin")
    return lda_model, lda_tfidf_model, kmeans_new, km_new, pca, scaler, ft_model

@st.cache_resource
def carga_3():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def lemmatize_stemming(text):
    ps = PorterStemmer()
    return ps.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def topicos(indic,datos):
    topics = []
    for y in range(datos.shape[0]):
        if len(indic[y]) > 0:
            valid_sublist = [sublist for sublist in indic[y] if len(sublist) > 1]
            if len(valid_sublist) > 0:
                max_index = np.argmax([sublist[1] for sublist in valid_sublist])
                topics.append(valid_sublist[max_index][0])
            else:
                topics.append(None)
        else:
            topics.append(None)
    return topics

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return cls_embedding.flatten()

def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,),dtype="float64")
    nwords = 0.
    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.get_word_vector(word))
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector

def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.words)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features)

def procesamiento(noticia):
    nuevos = pd.DataFrame({"headline_text":[noticia]})
    nuevos["errores"] = nuevos["headline_text"].apply(lambda x: tool.check(x))
    nuevos["corregido"] = nuevos.apply(lambda c: language_tool_python.utils.correct(c["headline_text"], c["errores"]), axis = 1)
    
    tokens = []
    for essay in nlp.pipe(nuevos['corregido'], batch_size=100):
        if essay.has_annotation("DEP"):
            tokens.append([e.text for e in essay])
        else:
            tokens.append(None) 
    nuevos['tokens'] = tokens

    nuevos['processed_text'] = nuevos.apply(lambda row:  ' '.join(token.lemma_ for token in nlp(row["corregido"]).sents), axis=1)
    nuevos['processed_text'] = nuevos['processed_text'].str.lower()
    nuevos['processed_text'] = nuevos['processed_text'].replace(list('áéíóú'),list('aeiou'),regex=True)
    nuevos['processed_text'] = nuevos['processed_text'].str.replace('[^\w\s]','')
    nuevos['processed_text'] = nuevos['processed_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_todas)]))
    processed_nuevo = nuevos['processed_text'].map(preprocess)
    bow_corpus_new = [dictionary.doc2bow(doc) for doc in processed_nuevo]
    return nuevos, bow_corpus_new

lda_model, lda_tfidf_model, kmeans_new, km_new, pca, scaler, ft_model = carga()
tool, nlp, stop_todas, dictionary, stop_words = carga_2()
tokenizer, model = carga_3()
nuevos = None

st.title("Clasificación de noticias según el titular")
repetida = ""
noticia = st.text_area("Ingrese el titular de la noticia", "Por favor que el titular sea en ingles, los métodos están entrenados en este idioma.")
metodo = st.selectbox("Seleccione el método para clasifica", ["","LDA sin TF-IDF","LDA con TF-IDF","Modelo Bert","Modelo Fast Text"])

if st.button("Clasificar noticia"):
    if noticia != repetida:
        nuevos, bow_corpus_new = procesamiento(noticia)
        repetida = noticia
    else: 
        pass
    
    temas = pd.DataFrame({"TEMA":["TEMA 1","TEMA 2", "TEMA 3"]})

    if metodo == "LDA sin TF-IDF":
        ind_wo_nuevo = lda_model[bow_corpus_new]
        topics_wo = topicos(ind_wo_nuevo, nuevos)
        resultado = temas["TEMA"][topics_wo[0]]
        st.success(f"El tema de la noticia es: {resultado}")

    elif metodo == "LDA con TF-IDF":
        temas = pd.DataFrame({"TEMA":["TEMA 3","TEMA 1", "TEMA 2"]})
        ind_tfidf_nuevo = lda_tfidf_model[bow_corpus_new]
        topics_tfidf = topicos(ind_tfidf_nuevo, nuevos)
        resultado = temas["TEMA"][topics_tfidf[0]]
        st.success(f"El tema de la noticia es: {resultado}")

    elif metodo == "Modelo Bert":
        nuevos['bert_embedding'] = nuevos['processed_text'].apply(get_bert_embedding)
        embeddings_nuevo = np.stack(nuevos['bert_embedding'].values)
        bert = kmeans_new.predict(embeddings_nuevo)
        resultado = temas["TEMA"][bert[0]]
        st.success(f"El tema de la noticia es: {resultado}")

    elif metodo == "Modelo Fast Text":
        ftext_nuevo = averaged_word_vectorizer(corpus=nuevos['tokens'], model=ft_model, num_features=ft_model.get_dimension())
        doc_embedding_new = pd.DataFrame(ftext_nuevo)
        pcs_new = pca.transform(doc_embedding_new)
        scaler_new = scaler.transform(pcs_new)
        fast = km_new.predict(scaler_new)
        resultado = temas["TEMA"][fast[0]]
        st.success(f"El tema de la noticia es: {resultado}")

    else : st.warning("Por favor elija un método")
