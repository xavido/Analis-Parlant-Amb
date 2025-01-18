import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from datetime import datetime

# Configuració inicial de NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

# Configuració de stopwords
stop_words_cat = {'que', 'què', 'com', 'quin', 'quina', 'quins', 'quines', 'per', 'els', 'les',
                 'un', 'una', 'uns', 'unes', 'el', 'la', 'els', 'les', 'i', 'o', 'però', 'perquè',
                 'si', 'no', 'hi', 'ha', 'té', 'tenen', 'tenim', 'està', 'estan', 'estem',
                 'aquest', 'aquesta', 'aquests', 'aquestes', 'aquell', 'aquella', 'aquells', 'aquelles','va','van','dels','amb','més','mes','és','són','eren'}

# Configuració de la pàgina
st.set_page_config(page_title="Anàlisi de Parlant Amb...", layout="wide")

# Títol principal
st.title("📊 Dashboard d'Anàlisi de Parlant Amb...")

# Configuració de la base de dades
DB_CONFIG =  st.secrets["DB_CONFIG"]

# Connexió a la base de dades
@st.cache_resource
def get_database_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        st.error(f"Error de connexió a la base de dades: {err}")
        return None

# Funció per carregar les dades
@st.cache_data
def load_data():
    conn = get_database_connection()
    if conn is None:
        st.stop()

    try:
        query = """
        SELECT id, pregunta, resposta, infografia, data, tema, idc, curso,topico
        FROM teclaPREGUNTES
        """
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error en carregar les dades: {str(e)}")
        return None
    finally:
        conn.close()

# Carregar dades
df = load_data()
if df is None:
    st.stop()

# Sidebar amb filtres
st.sidebar.title("Filtres")
selected_curso = st.sidebar.multiselect(
    "Filtra per curs:",
    options=sorted(df['curso'].unique()),
    default=sorted(df['curso'].unique())
)

selected_tema = st.sidebar.multiselect(
    "Filtra per tema:",
    options=sorted(df['topico'].unique()),
    default=sorted(df['topico'].unique())
)

# Aplicar filtres
filtered_df = df[
    (df['curso'].isin(selected_curso)) &
    (df['topico'].isin(selected_tema))
]

# Pestanyes principals
tab1, tab2, tab3 = st.tabs(["📈 Anàlisi Descriptiva", "📝 Anàlisi de Text", "👥 Anàlisi d'Usuaris"])

with tab1:
    st.header("Anàlisi Descriptiva")

    # Estadístiques bàsiques
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Usuaris Únics", len(filtered_df['idc'].unique()))

    with col2:
        avg_interactions = filtered_df.groupby('idc').size().mean()
        st.metric("Mitjana d'interaccions per usuari", f"{avg_interactions:.2f}")

    with col3:
        total_topics = len(filtered_df['topico'].unique())
        st.metric("Total de Temes", total_topics)

    with col4:
        images = filtered_df['infografia'].notna().sum()
        st.metric("Total d'Infografies", images)

    # Gràfics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribució de Temes")
        topic_dist = filtered_df['topico'].value_counts()
        fig = px.pie(
            values=topic_dist.values,
            names=topic_dist.index,
            title="Distribució de Temes"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribució per Curs")
        curso_dist = filtered_df['curso'].value_counts()
        fig = px.bar(
            x=curso_dist.index,
            y=curso_dist.values,
            title="Distribució per Curs"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Activitat temporal
    st.subheader("Activitat al llarg del temps")
    filtered_df['data'] = pd.to_datetime(filtered_df['data'])
    daily_activity = filtered_df.resample('D', on='data').size()
    fig = px.line(
        x=daily_activity.index,
        y=daily_activity.values,
        title="Activitat Diària",
        labels={'x': 'Data', 'y': 'Nombre d\'interaccions'}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Anàlisi de Text")

    # Preparar stopwords
    try:
        stop_words = set(stopwords.words('spanish') + stopwords.words('english'))
        stop_words.update(stop_words_cat)
    except Exception as e:
        stop_words = stop_words_cat
        st.warning("S'han carregat només les stopwords en català")

    # Anàlisi de freqüència de paraules
    st.subheader("Paraules més Freqüents en Preguntes")

    def get_word_freq(texts):
        words = []
        for text in texts:
            if isinstance(text, str):
                try:
                    tokens = word_tokenize(text.lower())
                    words.extend([w for w in tokens if w.isalnum() and w not in stop_words and len(w) > 1])
                except Exception as e:
                    continue
        return pd.Series(words).value_counts()

    word_freq = get_word_freq(filtered_df['pregunta'])

    # Mostrar núvol de paraules
    if not word_freq.empty:
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate_from_frequencies(word_freq)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error en generar el núvol de paraules: {str(e)}")

        # Mostrar les 20 paraules més freqüents
        st.subheader("Top 20 Paraules més Freqüents")
        fig = px.bar(
            x=word_freq.head(20).index,
            y=word_freq.head(20).values,
            title="Paraules més Freqüents en Preguntes"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Anàlisi de paraules per tema
    st.subheader("Anàlisi de Paraules per Tema")

    # Selector de tema
    selected_topic = st.selectbox(
        "Selecciona un tema per veure les seves paraules més freqüents:",
        options=sorted(filtered_df['topico'].unique())
    )

    # Filtrar preguntes pel tema seleccionat
    topic_questions = filtered_df[filtered_df['topico'] == selected_topic]['pregunta']
    topic_word_freq = get_word_freq(topic_questions)

    if not topic_word_freq.empty:
        col1, col2 = st.columns(2)

        with col1:
            try:
                # Núvol de paraules pel tema
                wordcloud_topic = WordCloud(width=400, height=300, background_color='white', max_words=50).generate_from_frequencies(topic_word_freq)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud_topic, interpolation='bilinear')
                ax.axis('off')
                plt.title(f"Núvol de paraules - {selected_topic}")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error en generar el núvol de paraules pel tema: {str(e)}")

        with col2:
            # Top 15 paraules més freqüents pel tema
            fig = px.bar(
                x=topic_word_freq.head(15).index,
                y=topic_word_freq.head(15).values,
                title=f"Top 15 Paraules més Freqüents - {selected_topic}"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Estadístiques del tema
        total_questions = len(topic_questions)
        avg_words = topic_questions.str.split().str.len().mean()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de preguntes", total_questions)
        with col2:
            st.metric("Mitjana de paraules per pregunta", f"{avg_words:.1f}")

    # Topic Modeling
    st.subheader("Descobriment de Temes")
    n_topics = st.slider("Nombre de temes a descobrir", min_value=2, max_value=10, value=5)

    try:
        vectorizer = CountVectorizer(max_features=1000, stop_words=list(stop_words))
        doc_term_matrix = vectorizer.fit_transform(filtered_df['pregunta'].fillna(''))

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_output = lda.fit_transform(doc_term_matrix)

        # Mostrar els temes descoberts
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10-1:-1]]
            st.write(f"Tema {topic_idx + 1}: {', '.join(top_words)}")
    except Exception as e:
        st.error(f"Error en el descobriment de temes: {str(e)}")

with tab3:
    st.header("Anàlisi de Comportament d'Usuaris")

    # Segmentació d'usuaris per freqüència d'ús
    user_activity = filtered_df.groupby('idc').size()

    def get_user_segment(interactions):
        if interactions <= 2:
            return "Poc actiu"
        elif interactions <= 5:
            return "Moderadament actiu"
        else:
            return "Molt actiu"

    user_segments = user_activity.apply(get_user_segment)
    segment_dist = user_segments.value_counts()

    st.subheader("Segmentació d'Usuaris")
    fig = px.pie(
        values=segment_dist.values,
        names=segment_dist.index,
        title="Distribució de Segments d'Usuaris"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Anàlisi de retenció
    st.subheader("Anàlisi de Retenció")

    user_first_interaction = filtered_df.groupby('idc')['data'].min()
    user_last_interaction = filtered_df.groupby('idc')['data'].max()

    retention_days = (user_last_interaction - user_first_interaction).dt.days

    fig = px.histogram(
        x=retention_days.values,
        nbins=30,
        title="Distribució del Temps de Retenció (dies)",
        labels={'x': 'Dies entre primera i última interacció', 'y': 'Nombre d\'usuaris'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Estadístiques de retenció
    st.write(f"Mitjana de dies de retenció: {retention_days.mean():.1f} dies")
    st.write(f"Mediana de dies de retenció: {retention_days.median():.1f} dies")
