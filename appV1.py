
import streamlit as st
from pymongo import MongoClient
from datetime import datetime, time
from datetime import date
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

mongo_uri = "mongodb+srv://matei21:matei21@cluster0.rdv7jrr.mongodb.net/"
client = MongoClient(mongo_uri)

db=client['movieDB']
collection = db['UniversalPictures']

API_KEY = "070b8a6943bc0ef5cbff411769b3b567"
def get_movie_poster(title):
    url = "https://api.themoviedb.org/3/search/movie"
    
    params = {
        "api_key": API_KEY,
        "query": title
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("results"):
        poster_path = data["results"][0].get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    
    return None

st.set_page_config(
    page_title="Proiect Dizertatie - Blegu-Duta Matei-Cosmin",
    page_icon=":chart:",
    layout="wide",
    initial_sidebar_state="auto"
)

st.logo("media/Universal-Logo.png", size = "large")

st.markdown('<h1>Dezvoltarea unei platforme web pentru analiza performanței filmelor produse de studioul Universal Pictures<h1>', unsafe_allow_html=True)

if 'db_version' not in st.session_state:
    st.session_state['db_version'] = 1

@st.cache_data(ttl=3600)
def get_movies_list():
    return [m["movieTitle"] for m in collection.find({}, {"movieTitle": 1})]

def mpaa_rating_logo(rating):
    if rating == "G":
        st.image(r"media/MPAA Logos/Mpaagrating2000svector_(white).svg")
    if rating == "PG":
        st.image(r"media/MPAA Logos/Mpaapgrating2000svector2_(white).svg")
    if rating == "PG-13":
        st.image(r"media/MPAA Logos/Mpaapg13rating2000svector_(white).svg")
    if rating == "R":
        st.image(r"media/MPAA Logos/Mpaarrating2000svector_(white).svg")
    if rating == "NC-17":
        st.image(r"media/MPAA Logos/Mpaanc17rating2000svector_(white).svg")

def sidebar_navigation():
    st.sidebar.markdown("# Segmente proiect")

    sections = [
        "Vizualizare si manipulare date",
        "Analiza Partea 1 - Analiza Valori Numerice",
        "Analiza Partea 2 - Analiza Valori Categorice",
        "Analiza Partea 3 - Analiza Predictiva"
    ]

    selected = st.sidebar.radio("Selecteaza segmentul:", sections)
    return selected

def manipulare_date():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Vizualizare Filme", "Adauga Film", "Sterge Film", "Editeaza Film", "Informatii Dataset"])

    with tab2:
        st.header("Adauga Film")
        create_movie_title = st.text_input("Titlu Film", key="create_movie_title")
        create_movie_release_date = st.date_input("Data Lansarii", key="create_movie_release_date")
        converted_release_date = datetime.combine(create_movie_release_date, time())
        create_movie_runtime = st.number_input("Durata Film", key="create_movie_runtime", step=1)
        create_movie_budget = st.number_input("Buget (USD)", key="create_movie_budget",step=1)
        create_movie_box_office = st.number_input("Box Office (USD)", key="create_movie_box_office",step=1)
        create_movie_genres = st.text_input("Genuri", key="create_movie_genres")
        create_movie_cast = st.text_input("Actori Principali", key="create_movie_cast")
        create_movie_age_rating = st.selectbox("Clasificare de Varsa MPAA", ("G", "PG", "PG-13", "R", "NC-17"),index = None, key="create_movie_age_rating")
        create_movie_box_office_domestic = st.number_input("Box Office Domestic (USD)", key="create_movie_box_office_dom",step=1)
        create_movie_box_office_international = st.number_input("Box Office International (USD)", key="create_movie_box_office_int",step=1)
        
        franchise_list = collection.distinct("franchise")
        franchise_list.append("Other")
        create_movie_franchise = st.selectbox("Franciza Film", franchise_list, index = 0, key="create_movie_franchise")
        selected_movie_franchise = " "
        if create_movie_franchise == "Other":
            create_new_movie_franchise = st.text_input("Insereaza Franciza Noua", key="create_new_movie_franchse")
            if (create_new_movie_franchise):
                selected_movie_franchise = create_new_movie_franchise.strip()
            else:
                selected_movie_franchise = "No Franchise"
        else:
            selected_movie_franchise = create_movie_franchise

        if st.button("Adauga Film"):
            if create_movie_title:
                genres_list = [g.strip() for g in create_movie_genres.split(",")] if create_movie_genres else []
                cast_list = [c.strip() for c in create_movie_cast.split(",")] if create_movie_cast else []
                movie_poster = get_movie_poster(create_movie_title)
                
                new_movie = {
                    "movieTitle": create_movie_title,
                    "releaseDate": converted_release_date,
                    "runTime": create_movie_runtime,
                    "budget": create_movie_budget,
                    "boxOffice": create_movie_box_office,
                    "genres": genres_list,             
                    "top_cast": cast_list,
                    "age_rating": create_movie_age_rating,
                    "boxOffice_domestic": create_movie_box_office_domestic,
                    "boxOffice_international": create_movie_box_office_international,
                    "franchise" : selected_movie_franchise,
                    "moviePoster": movie_poster
                }
                
                try:
                    collection.insert_one(new_movie)
                    st.success(f"Filmul '{create_movie_title}' a fost adaugat cu succes!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Eroare la inserare: {e}")
            else:
                st.warning("Te rog introdu cel putin titlul filmului.")

    with tab1:
        st.header("Vizualizare Filme")
        list_genres = collection.distinct("genres")
        list_ratings = collection.distinct("age_rating")
        list_cast = collection.distinct("top_cast")
        stats = list(collection.aggregate([{"$group": {"_id": None,"min_budget": {"$min": "$budget"},"max_budget": {"$max": "$budget"},
                                                       "min_box_office_total": {"$min": "$boxOffice"},"max_box_office_total": {"$max": "$boxOffice"},
                                                       "min_box_office_int": {"$min": "$boxOffice_international"},"max_box_office_int": {"$max": "$boxOffice_international"},
                                                       "min_box_office_dom": {"$min": "$boxOffice_domestic"},"max_box_office_dom": {"$max": "$boxOffice_domestic"},
                                                       "min_date": {"$min": "$releaseDate"},"max_date": {"$max": "$releaseDate"},
                                                       "min_runtime": {"$min": "$runTime"},"max_runtime": {"$max": "$runTime"}}}]))
        if stats:
            min_db_budget = stats[0]["min_budget"]
            max_db_budget = stats[0]["max_budget"]
            initial_range_budget = (min_db_budget, max_db_budget)

            min_db_box_office_total = stats[0]["min_box_office_total"]
            max_db_box_office_total = stats[0]["max_box_office_total"]
            initial_range_box_office_total = (min_db_box_office_total, max_db_box_office_total)

            min_db_box_office_int = stats[0]["min_box_office_int"]
            max_db_box_office_int = stats[0]["max_box_office_int"]
            initial_range_box_office_int = (min_db_box_office_int, max_db_box_office_int)

            min_db_box_office_dom = stats[0]["min_box_office_dom"]
            max_db_box_office_dom = stats[0]["max_box_office_dom"]
            initial_range_box_office_dom = (min_db_box_office_dom, max_db_box_office_dom)

            min_release_date = stats[0]["min_date"]
            max_release_date = stats[0]["max_date"]
            initial_range_release_date = (min_release_date, max_release_date)

            min_db_runtime = stats[0]["min_runtime"]
            max_db_runtime = stats[0]["max_runtime"]
            initial_range_runtime = (min_db_runtime, max_db_runtime)


        with st.expander("Filtrare Filme"):
                col_filter_genre, col_filter_franchise, col_filter_ratings, col_filter_cast = st.columns(4)
                with col_filter_genre:
                    select_filter_genre = st.multiselect("Selecteaza Genul(urile)", list_genres, key="select_genre_filter")
                with col_filter_franchise:
                    select_filter_franchise = st.multiselect("Selecteaza Franciza(ele)", franchise_list, key="select_franchise_filter")
                with col_filter_ratings:
                    select_filter_ratings = st.multiselect("Selecteaza Rating-ul MPAA", list_ratings, key="select_rating_filter")
                with col_filter_cast:
                    select_filter_cast = st.multiselect("Selecteaza Actorul(ii)", list_cast, key="select_cast_filter")
                    
                col_filter_budget, col_filter_box_total = st.columns(2)
                with col_filter_budget:
                    select_filter_budget = st.slider("Selecteaza Bugetul", min_db_budget, max_db_budget, initial_range_budget, step=50000000 ,format="$%d",key="filter_budget")
                    budget_min_ales, budget_max_ales = select_filter_budget
                with col_filter_box_total:
                    select_filter_box_total = st.slider("Selecteaza box office-ul total", min_db_box_office_total, max_db_box_office_total, initial_range_box_office_total, step=50000000 ,format="$%d", key="range_box")
                    min_db_box_office_total, max_db_box_office_total = select_filter_box_total
                
                with st.expander("Filtrare dupa incasari domestice si internationale"):
                    col_filter_box_dom, col_filter_box_int = st.columns(2)
                    with col_filter_box_dom:
                        select_filter_box_dom = st.slider("Selecteaza box office-ul domestic", min_db_box_office_dom, max_db_box_office_dom, initial_range_box_office_dom, step=50000000 ,format="$%d", key="range_box_dom")
                        min_db_box_office_dom, max_db_box_office_dom = select_filter_box_dom
                    with col_filter_box_int:
                        select_filter_box_int = st.slider("Selecteaza box office-ul international", min_db_box_office_int, max_db_box_office_int, initial_range_box_office_int, step=50000000 ,format="$%d", key="range_box_int")
                        min_db_box_office_int, max_db_box_office_int = select_filter_box_int
                
                col_filter_release_date, col_filter_runtime = st.columns(2)
                with col_filter_release_date:
                    select_filter_date = st.slider("Selecteaza Data Lansarii", min_release_date, max_release_date, initial_range_release_date,key="filter_release_date")
                    min_release_date, max_release_date = select_filter_date
                with col_filter_runtime:
                    select_filter_runtime = st.slider("Selecteaza Durata", min_db_runtime, max_db_runtime, initial_range_runtime,format="%d minute", key="filter_runtime")
                    min_db_runtime, max_db_runtime = select_filter_runtime

        mongo_query = {}
        if (select_filter_genre):
            mongo_query["genres"] = {"$in": select_filter_genre}
        if (select_filter_franchise):
            mongo_query["franchise"] = {"$in": select_filter_franchise}
        if (select_filter_ratings):
            mongo_query["age_rating"] = {"$in": select_filter_ratings}
        if (select_filter_cast):
            mongo_query["top_cast"] = {"$in": select_filter_cast}
        if (select_filter_budget):
            mongo_query["budget"] = {"$gte": budget_min_ales, "$lte": budget_max_ales}
        if (select_filter_box_total):
            mongo_query["boxOffice"] = {"$gte": min_db_box_office_total, "$lte": max_db_box_office_total}
        if (select_filter_box_dom):
            mongo_query["boxOffice_domestic"] = {"$gte": min_db_box_office_dom, "$lte": max_db_box_office_dom}
        if (select_filter_box_int):
            mongo_query["boxOffice_international"] = {"$gte": min_db_box_office_int, "$lte": max_db_box_office_int}
        if (select_filter_date):
            mongo_query["releaseDate"] = {"$gte": min_release_date, "$lte": max_release_date}
        if (select_filter_runtime):
            mongo_query["runTime"] = {"$gte": min_db_runtime, "$lte": max_db_runtime}
            
        # if st.button("Look up movies in the MongoDB database"):
        #st.cache_data.clear()
        movies = collection.find(mongo_query)
        for movie in movies:
            m_id = str(movie.get("_id"))
            with st.container(border=True):
                col1, col2 = st.columns([1, 2])

                with col1:
                    if movie.get("moviePoster"):
                        st.image(movie.get("moviePoster"), use_container_width=True)

                with col2:
                    st.subheader(movie.get("movieTitle"))
                    c_franchise, c_rating = st.columns(2)
                    with c_franchise:
                        st.text_input("Franciza Film", value=movie.get("franchise"), key=f"franchise_{m_id}_{st.session_state['db_version']}")
                    with c_rating:
                        st.markdown('<p style="font-size: 0.85rem; margin-bottom: 0px;">Clasificare MPAA</p>', unsafe_allow_html=True)
                        mpaa_rating_logo(movie.get("age_rating"))

                    c_date, c_run = st.columns(2)
                    with c_date:
                        st.date_input("Data Lansarii", value=movie.get("releaseDate"), key=f"date_{m_id}_{st.session_state['db_version']}")
                    with c_run:
                        st.number_input("Durata Film", value=movie.get("runTime"), key=f"run_{m_id}_{st.session_state['db_version']}")

                    c_box_total, c_budget = st.columns(2)
                    with c_box_total:
                        if (movie.get("boxOffice")):
                            st.number_input("Box Office - Total", value=movie.get("boxOffice"), key=f"box_{m_id}_{st.session_state['db_version']}")
                        else:
                            st.text_input("Box Office - Total", value="Lansat exclusiv pe streaming - Fara date despre Box Office", key=f"box_2{m_id}_{st.session_state['db_version']}")
                    with c_budget:
                        if (movie.get("budget")):
                            st.number_input("Buget", value=movie.get("budget"), key=f"budget_{m_id}_{st.session_state['db_version']}")
                        else:
                            st.text_input("Buget", value="Fara date despre buget", key=f"budget_2{m_id}_{st.session_state['db_version']}")
                    
                    if (movie.get("boxOffice")):
                        c_box_off_dom, c_box_off_int = st.columns(2)
                        with c_box_off_dom:
                            if (movie.get("boxOffice_domestic")):
                                st.number_input("Box Office - Domestic", value=movie.get("boxOffice_domestic"), key=f"box_dom_{m_id}_{st.session_state['db_version']}")
                            else: 
                                st.text_input("Box Office - Domestic", value="Lansat exclusiv pe streaming in SUA",key=f"box_dom{m_id}_{st.session_state['db_version']}")
                        with c_box_off_int:
                            if (movie.get("boxOffice_international")):
                                st.number_input("Box Office - International", value=movie.get("boxOffice_international"), key=f"box_dom_2{m_id}_{st.session_state['db_version']}")
                            else:
                                st.text_input("Box Office - International", value="Lansat exclusiv pe streaming in afara SUA", key=f"box_int_2{m_id}_{st.session_state['db_version']}")
                    genres = movie.get("genres", [])
                    if genres:
                        st.text_input("Genuri", f"{' | '.join(genres)}", key=f"genres_{m_id}_{st.session_state['db_version']}")
                    cast = movie.get("top_cast", [])
                    if cast:
                        st.text_input("Distributie", f"{' | '.join(cast)}", key=f"cast_{m_id}_{st.session_state['db_version']}")
                    # st.text_input("MPAA Rating", value=movie.get("age_rating"), key=f"age_rating_{m_id}")
                    # mpaa_rating_logo(movie.get("age_rating"))
            
    with tab3:
        st.header("Sterge Film")
        movies = get_movies_list()
        movie_to_delete = st.selectbox("Selecteaza Filmul de Sters", options=movies,index=None)
        if st.button("Sterge Film"):
            query = {"movieTitle": movie_to_delete}
            result = collection.delete_one(query)
            st.cache_data.clear()
            if result.deleted_count > 0:
                st.success(f"Filmul {movie_to_delete} a fost sters")
            else:
                st.warning("error deleting")

    with tab4:
        st.header("Editeaza Film")
        movies = get_movies_list()
        movie_to_edit = st.selectbox("Selecteaza Filmul de Editat", options=movies,index=None)
        if movie_to_edit:
            with st.spinner("Information loading"):
                editable_movie = collection.find_one({"movieTitle": movie_to_edit},{})
                
                current_franchise = editable_movie.get("franchise")
                if current_franchise in franchise_list:
                    current_franchise_index = franchise_list.index(current_franchise)
                else:
                    current_franchise_index = 0
            #edit_movie_franchise = st.selectbox("Franchise", options=franchise_list, index=current_franchise_index)
            with st.form("edit_form"):
                edit_movie_title = st.text_input("Titlu Film", value= editable_movie.get("movieTitle"), key=f"edit_title_{movie_to_edit}_{st.session_state['db_version']}")
                edit_movie_franchise = st.selectbox("Franciza", options=franchise_list, index=current_franchise_index, key=f"edit_franchise_{movie_to_edit}_{st.session_state['db_version']}")
                create_new_movie_franchise = st.text_input("Insereaza Franciza Noua", key=f"create_new_movie_franchse_edit_{movie_to_edit}_{st.session_state['db_version']}")
                edit_movie_release_date = st.date_input("Data Lansarii",value= editable_movie.get("releaseDate"),key=f"edit_date_{movie_to_edit}_{st.session_state['db_version']}")
                edit_release_date = datetime.combine(edit_movie_release_date, time())
                edit_movie_runtime = st.number_input("Durata Film (minute)", step=1, value= editable_movie.get("runTime"),key=f"edit_time_{movie_to_edit}_{st.session_state['db_version']}")
                edit_movie_box_office = st.number_input("Box Office - Total", value= editable_movie.get("boxOffice"), step=1,key=f"edit_boxT_{movie_to_edit}_{st.session_state['db_version']}")
                edit_movie_box_office_dom = st.number_input("Box Office - Domestic", value= editable_movie.get("boxOffice_domestic"), step=1, key=f"edit_boxD_{movie_to_edit}_{st.session_state['db_version']}")
                edit_movie_box_office_int = st.number_input("Box Office - International", value= editable_movie.get("boxOffice_international"), step=1, key=f"edit_BoxI_{movie_to_edit}_{st.session_state['db_version']}")
                edit_movie_budget = st.number_input("Buget", value= editable_movie.get("budget"), step=1,key=f"edit_budget_{movie_to_edit}_{st.session_state['db_version']}")
                current_genres = ", ".join(editable_movie.get("genres", []))
                edit_movie_genres = st.text_input("Genuri", value=current_genres, key=f"edit_genre_{movie_to_edit}_{st.session_state['db_version']}")
                current_cast = ", ".join(editable_movie.get("top_cast", []))
                edit_movie_cast = st.text_input("Actori Principali", value=current_cast, key=f"edit_cast_{movie_to_edit}_{st.session_state['db_version']}")
                mpaa_ratings = ("G", "PG", "PG-13", "R", "NC-17")
                current_rating = editable_movie.get("age_rating")
                if current_rating in mpaa_ratings:
                    current_rating_index = mpaa_ratings.index(current_rating)
                else:
                    current_rating_index = 0
                edit_movie_age_rating = st.selectbox("Rating MPAA", options=mpaa_ratings, index=current_rating_index,key=f"edit_rating_{movie_to_edit}_{st.session_state['db_version']}")

                if st.form_submit_button("Editeaza Film"):
                    selected_movie_franchise = " "
                    if edit_movie_franchise == "Other":
                        if (create_new_movie_franchise):
                            selected_movie_franchise = create_new_movie_franchise.strip()
                        else:
                            selected_movie_franchise = "No Franchise"
                    else:
                        selected_movie_franchise = edit_movie_franchise
                    updated_genres = [g.strip() for g in edit_movie_genres.split(",")] if edit_movie_genres else []
                    updated_cast = [g.strip() for g in edit_movie_cast.split(",")] if edit_movie_cast else []
                    query = {"movieTitle": movie_to_edit} 
                    update = {"$set" : {"movieTitle" : edit_movie_title, "releaseDate" : edit_release_date, 
                                        "runTime" : edit_movie_runtime, "boxOffice": edit_movie_box_office, 
                                        "budget": edit_movie_budget, "genres": updated_genres, "top_cast": updated_cast,
                                        "age_rating": edit_movie_age_rating, "boxOffice_domestic": edit_movie_box_office_dom,
                                        "boxOffice_international": edit_movie_box_office_int, "franchise": selected_movie_franchise,
                                        "moviePoster": get_movie_poster(edit_movie_title)}}
                    st.cache_data.clear()
                    result = collection.update_one(query, update)
                    if result.modified_count > 0:
                        st.success(f"Filmul {movie_to_edit} a fost editat")
                        st.session_state['db_version'] += 1
                        st.rerun()
                    else:
                        st.warning("error editing")

    with tab5:
        st.title("Analiza Date Din Dataset")
    
        try:
            cursor = collection.find()
            cursor_data = list(cursor)
            if cursor_data:
                df_new = pd.DataFrame(cursor_data)
                st.session_state['df_shared'] = df_new   
        except Exception as e:
            st.error(f"Eroare: {e}")

        if "df_shared" in st.session_state:
            df = st.session_state['df_shared']
            st.header("Salveaza datele in CSV")
            columns_to_download = st.multiselect("Selecteaza coloanele pentru descarcare", df.columns)
            df_to_download = df[columns_to_download]
            csv = df_to_download.to_csv(index=False).encode('utf-8')
            #date_of_dl = date.today()
            st.download_button("Descarca CSV", csv, f"Date_Universal_Pictures_{date.today()}.csv", "secondary")
            st.header("Informatii dataset")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Numar de filme in baza de date:", len(df))
            with col2:
                avg_run = np.mean(df['runTime']) if 'runTime' in df.columns else 0
                st.metric("Durata medie per film", f"{avg_run:.1f} min")
            with col3:
                avg_box = np.mean(df['boxOffice']) if 'boxOffice' in df.columns else 0
                st.metric("Incasari medii per film", f"${avg_box:,.0f} (USD)")
            with col4:
                avg_budget = np.mean(df['budget']) if 'budget' in df.columns else 0
                st.metric("Buget mediu per film", f"${avg_budget:,.0f} (USD)")
            
            n_rows = st.slider("Numar randuri de afisat:", min_value = 3, max_value = len(df), value = len(df), key="afisat_numar_randuri")
            st.dataframe(df.head(n_rows), use_container_width=True)
            with st.expander("Ultimele Randuri"):
                st.dataframe(df.tail(n_rows), use_container_width=True)
            
            st.header("Buget contra incasari per film")
            list_titles = collection.distinct("movieTitle")
            titlu_ptr_bar = st.multiselect("Selecteaza filmul pentru comparatie financiara",list_titles, key="char")
            df_film_selectat = df[df["movieTitle"].isin(titlu_ptr_bar)]
            fig_box1 = px.bar(
                df_film_selectat,
                title=f"Comparatie Financiara Pentru: {titlu_ptr_bar}",
                x = "movieTitle",
                y = ["budget","boxOffice","boxOffice_domestic","boxOffice_international"],
                labels={
                    "movieTitle" : "Titlu Film",
                    "value" : "Milioane (USD)",
                },
                barmode="group"
            )
            fig_box1.update_layout(title_font_size=25)
            st.plotly_chart(fig_box1,width='stretch')
            
            st.header("Informatii despre campurile din baza de date")
            c1, c2 = st.columns(2)
            with c1:
                dtype_df = pd.DataFrame({
                    "Coloana": df.columns,
                    "Tip": df.dtypes.astype(str),
                    "Non-Null": df.count().values,
                    "Null": df.isnull().sum().values
                })
                st.dataframe(dtype_df, use_container_width=True)
            with c2:
                type_counts = df.dtypes.astype(str).value_counts()
                fig = px.pie(values=type_counts.values, names=type_counts.index, title="Tipuri de Date")
                st.plotly_chart(fig, use_container_width=True)

            st.header("Informatii despre campurile numerice din baza de date")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
def analiza_date_partea_1():
    st.title("Analiza Valori Numerice")
    if 'df_shared' not in st.session_state:
        st.warning("Eroare la conectarea cu baza de date.")
        return

    df = st.session_state['df_shared'].copy()
    df['releaseDate'] = pd.to_datetime(df['releaseDate'])
    df['year'] = df['releaseDate'].dt.year
    df['month_name'] = df['releaseDate'].dt.month_name()

    if '_id' in df.columns:
        df['_id'] = df['_id'].astype(str)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("Nu exista coloane numerice")
        return

    col_for_hist = st.selectbox("Selecteaza coloana pentru histograma:", numeric_cols, key="hist_col")
    n_bins = st.slider("Numar de bins pentru histograma:", 10, 100, 50, 1, key="bins_slider")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.histogram(
            df,
            x=col_for_hist,
            nbins=n_bins,
            title=f'Histograma: {col_for_hist}',
            marginal='box'
        )
        fig.update_layout(title_font_size=25)
        st.plotly_chart(fig,width='stretch')

    with col2:
        stats_df = pd.DataFrame({
            'Metrica': ['Minim', 'Q1 (25%)', 'Mediana', 'Q3 (75%)', 'Maxim', 'Media', 'Std Dev'],
            'Valoare': [
                df[col_for_hist].min(),
                df[col_for_hist].quantile(0.25),
                df[col_for_hist].median(),
                df[col_for_hist].quantile(0.75),
                df[col_for_hist].max(),
                df[col_for_hist].mean(),
                df[col_for_hist].std()
            ]
        })
        st.dataframe(stats_df, use_container_width=True)
    
    st.divider()
    df_bk = df.drop(columns=['_id', 'moviePoster', 'movieTitle', 'releaseDate'])
    cat_cols = df_bk.select_dtypes(include=['object']).columns.tolist()
    
    for extra_col in ['genres', 'top_cast','year']:
        if extra_col in df_bk.columns and extra_col not in cat_cols:
            cat_cols.append(extra_col)
    cat_col = st.selectbox("Selecteaza Categoria pentru Informatii Financiara:", cat_cols, key="cat_col2")
    df_bar_cat = df_bk.explode(cat_col)
    df_grouped = df_bar_cat.groupby(cat_col)[["budget", "boxOffice", "boxOffice_domestic", "boxOffice_international"]].sum().reset_index()
    fig_box2 = px.bar(
        df_grouped,
        title=f"Informatii Financiare Pentru: {cat_col}",
        x = cat_col,
        y = ["budget","boxOffice","boxOffice_domestic","boxOffice_international"],
        labels={
            "value" : "Milioane (USD)",
        },
        barmode="group"
    )
    fig_box2.update_layout(title_font_size=25)
    st.plotly_chart(fig_box2,use_container_width=True)

    corr_method = st.radio(
        "Metoda de corelatie:",
        ['pearson', 'spearman', 'kendall'],
        format_func=lambda x: {
            'pearson': 'Pearson (Linear)',
            'spearman': 'Spearman (Rank)',
            'kendall': 'Kendall (Rank)'
        }[x],
        horizontal=True
    )

    corr_matrix = df[numeric_cols].corr(corr_method)

    fig_hm = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title=f"Heatmap Corelatie ({corr_method.capitalize()})"
    )
    fig_hm.update_xaxes(tickangle=45)
    fig_hm.update_layout(title_font_size=25)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("Variabila X:", numeric_cols, key="scatter_x")
    with c2:
        y_options = [c for c in numeric_cols if c != x_col]
        y_col = st.selectbox("Variabila Y:", y_options, key="scatter_y")

    fig_sc = px.scatter(df, x=x_col, y=y_col, title=f"Scatter: {x_col} vs {y_col}",hover_name="movieTitle")
    fig_sc.update_layout(title_font_size=25)
    st.plotly_chart(fig_sc, use_container_width=True)

    pearson_r = df[[x_col, y_col]].dropna().corr(method="pearson").iloc[0, 1]
    st.metric("Coeficient Pearson r", "N/A" if pd.isna(pearson_r) else f"{pearson_r:.4f}")

    st.divider()

    def iqr_bounds(series, k=1.5):
        s = series.dropna()
        if s.empty:
            return None, None
        else:
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            return q1 - k * iqr, q3 + k * iqr

    def iqr_outlier_mask(df_, col, k = 1.5):
        low, high = iqr_bounds(df_[col], k=k)
        if low is None or high is None:
            return pd.Series([False] * len(df_), index=df_.index)
        return (df_[col] < low) | (df_[col] > high)

    k = st.slider("Factor IQR (k):", 1.0, 3.0, 1.5, 0.1, key="iqr_k")

    rows = []
    for col in numeric_cols:
        mask = iqr_outlier_mask(df, col, k=k)
        n_out = int(mask.sum())
        pct_out = (n_out / len(df) * 100)
        low, high = iqr_bounds(df[col], k=k)
        rows.append({
            "Coloana": col,
            "Outlieri (nr)": n_out,
            "Outlieri (%)": round(pct_out, 2),
            "Lower fence": None if low is None else float(low),
            "Upper fence": None if high is None else float(high),
        })

    outlier_summary = pd.DataFrame(rows).sort_values("Outlieri (nr)", ascending=False)
    st.dataframe(outlier_summary, use_container_width=True)

    col_for_box = st.selectbox("Selecteaza coloana pentru box plot:", numeric_cols, key="box_col")

    low, high = iqr_bounds(df[col_for_box], k=k)
    mask_box = iqr_outlier_mask(df, col_for_box, k=k)
    n_outliers = int(mask_box.sum())
    pct_outliers = (n_outliers / len(df) * 100)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total valori", f"{len(df):,}")
    with m2:
        st.metric("Outlieri gasiti", f"{n_outliers:,}")
    with m3:
        st.metric("Procent outlieri", f"{pct_outliers:.2f}%")

    fig_box1 = px.box(df, y=col_for_box, points="outliers", title=f"Box Plot (IQR k={k}): {col_for_box}", hover_name="movieTitle")
    if low is not None and high is not None:
        fig_box1.add_hline(y=low, line_dash="dash", line_color="red", annotation_text="Lower Fence")
        fig_box1.add_hline(y=high, line_dash="dash", line_color="red", annotation_text="Upper Fence")
    fig_box1.update_layout(title_font_size=25)
    st.plotly_chart(fig_box1, use_container_width=True)


    st.header("Sectiunea Altair")
    heatmap = alt.Chart(df,title=alt.TitleParams(text='Distributie de filme per an per luna', fontSize=30)).mark_rect().encode(
    x=alt.X('month_name:O', title='Luna Lansarii', sort=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']),
    y=alt.Y('year:O', title='Anul Lansarii'),
    color=alt.Color('sum(boxOffice):Q', title='Total Box Office', scale=alt.Scale(scheme='turbo')),
    tooltip=['year', 'month_name', 'sum(boxOffice):Q', 'count(movieTitle):Q'])
    st.altair_chart(heatmap, use_container_width=True)

    st.divider()
    df_explodat = df.explode('genres')
    click_genres = alt.selection_point(fields=['genres'])
    chart_master = alt.Chart(df_explodat,title=alt.TitleParams(text='Genuri', fontSize=30)).mark_bar().encode(
        x=alt.X('genres:N', title='Genuri de Film', sort='-y'),
        y=alt.Y('sum(boxOffice):Q', title='Box Office Total (USD)', axis=alt.Axis(format='$~s')),
        color=alt.condition(click_genres, alt.value('#1f77b4'), alt.value('lightgray')),
        tooltip=['genres', 'sum(boxOffice):Q', 'count(movieTitle):Q']).add_params(click_genres)
    chart_detail = alt.Chart(df_explodat,title=alt.TitleParams(text='Box Office Per Gen Per Film', fontSize=30)).mark_bar(invalid=None).encode(
        x=alt.X('movieTitle:N', title='Film in Gen', sort='-y'),
        y=alt.Y('boxOffice:Q', title='Box Office', axis=alt.Axis(format='$~s')),
        color=alt.Color('age_rating:N', title='Rating MPAA'),
        tooltip=['movieTitle', 'budget', 'boxOffice', 'releaseDate']).transform_filter(click_genres)
    st.altair_chart(chart_master & chart_detail, use_container_width=True)

    st.divider()
    brush = alt.selection_interval()
    base = alt.Chart(df,title=alt.TitleParams(text='Performanta Financiara a Filmelor', fontSize=30)).encode(
        x=alt.X('budget:Q', title='Buget (USD)', axis=alt.Axis(format='$~s')),
        y=alt.Y('boxOffice:Q', title='Box Office (USD)', axis=alt.Axis(format='$~s')),
        tooltip=['movieTitle', 'budget', 'boxOffice'])
    scatter = base.mark_circle(size=100).encode(color=alt.condition(brush, alt.Color('movieTitle:N',legend=None) , alt.value('lightgray'))).add_params(brush)
    line_budget = alt.Chart(df).mark_rule(color='red', strokeDash=[4, 4]).encode(x='mean(budget):Q').transform_filter(brush)
    line_box_office = alt.Chart(df).mark_rule(color='red', strokeDash=[4, 4]).encode(y='mean(boxOffice):Q').transform_filter(brush)
    quadrant_chart = alt.layer(scatter, line_budget, line_box_office)
    st.altair_chart(quadrant_chart, use_container_width=True)

def analiza_date_partea_2():
    st.title("Analiza Valori Categorice")
    if 'df_shared' not in st.session_state:
        st.warning("Eroare la conectarea cu baza de date.")
        return

    df = st.session_state['df_shared'].copy()
    df['releaseDate'] = pd.to_datetime(df['releaseDate'])
    df['year'] = df['releaseDate'].dt.year
    # df['month'] = df['releaseDate'].dt.month
    df['month_name'] = df['releaseDate'].dt.month_name()
    
    if '_id' in df.columns:
        df['_id'] = df['_id'].astype(str)

    df_bk_2 = df.drop(columns=['_id', 'moviePoster', 'movieTitle', 'releaseDate'])
    cat_cols = df_bk_2.select_dtypes(include=['object']).columns.tolist()
    
    for extra_col in ['genres', 'top_cast','year']:
        if extra_col in df_bk_2.columns and extra_col not in cat_cols:
            cat_cols.append(extra_col)

    if not cat_cols:
        st.error("Nu exista coloane categorice in dataset!")
        return

    cat_col = st.selectbox("Selecteaza coloana categorica:", cat_cols, key="cat_col")
    
    non_null_series = df[cat_col].dropna()
    if not non_null_series.empty and isinstance(non_null_series.iloc[0], list):
        df_final = df.explode(cat_col)
    else:
        df_final = df

    value_counts = df_final[cat_col].value_counts()
    value_counts_pct = (value_counts / len(df_final) * 100).round(2)

    fig_tree = px.treemap(
        df_final, 
        path=[cat_col], 
        title=f'Treemap-ul Categoriilor: {cat_col}',
        color=cat_col
    )
    fig_tree.update_layout(title_font_size=25)
    st.plotly_chart(fig_tree, use_container_width=True)

    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        labels={'x': cat_col, 'y': 'Frecventa'},
        title=f'Distributia Categoriilor: {cat_col}',
        text=value_counts.values
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(title_font_size=25)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1: 
        freq_df = pd.DataFrame({
            'Categorie': value_counts.index,
            'Frecventa': value_counts.values,
            'Procent': value_counts_pct.values
        })
        st.dataframe(freq_df, use_container_width=True)

    with col2: 
        fig_pie = px.pie(
            names=value_counts.index, 
            values=value_counts.values, 
            title=f'Distributia Procentuala: {cat_col}',
            hole=0.4 
        )
        fig_pie.update_layout(title_font_size=25)
        st.plotly_chart(fig_pie, use_container_width=True)

    numeric_cols_box = df.drop(["year"], axis =1).select_dtypes(include=[np.number]).columns.tolist()
    col_numerica1 = st.selectbox("Selecteaza coloana numerica pentru box plot:", numeric_cols_box, key="box_plot_col")

    # n_rows = st.slider("Numar randuri de afisat:", min_value = 2, max_value = len(df_final), value = len(df_final), key="afisat_numar_randuri")
    st.divider()
    fig_box2 = px.box(
        df_final, #.head(n_rows), 
        x=cat_col, 
        y=col_numerica1, 
        title=f'{col_numerica1}-ul filmului in functie de {cat_col}',
        color=cat_col
    )
    fig_box2.update_layout(title_font_size=25)
    st.plotly_chart(fig_box2, use_container_width=True, key="box_plot1")

    col_numerica2 = st.selectbox("Selecteaza coloana numerica pentru al 2lea box plot:", numeric_cols_box, key="box_plot_col2")
    st.divider()
    fig_box3 = px.box(
        df_final, 
        x=cat_col, 
        y=col_numerica2, 
        title=f'{col_numerica2}-ul filmului in functie de {cat_col}',
        color=cat_col
    )
    fig_box3.update_layout(title_font_size=25)
    st.plotly_chart(fig_box3, use_container_width=True,key="box_plot2")

    yearly_data = df.groupby('year')[['boxOffice', 'budget']].sum().reset_index()
    fig = px.line(yearly_data, x='year', y=['boxOffice','budget'], title="Evolutia incasarilor pe Ani")
    fig.update_layout(title_font_size=25)
    st.plotly_chart(fig)

def machine_learning_partea_3():
    st.title("Analiza Predictiva")
    if 'df_shared' not in st.session_state:
        st.warning("Eroare la conectarea cu baza de date.")
        return

    df = st.session_state['df_shared'].copy()

    df['releaseDate'] = pd.to_datetime(df['releaseDate'])
    df['year'] = df['releaseDate'].dt.year
    df['month_name'] = df['releaseDate'].dt.month_name()
    df_ml = df.drop(columns=['_id', 'moviePoster', 'movieTitle', 'releaseDate'])
    cat_cols = df_ml.select_dtypes(include=['object']).columns.tolist()
    for extra_col in ['genres', 'top_cast','year','month_name']:
        if extra_col in df_ml.columns and extra_col not in cat_cols:
            cat_cols.append(extra_col)

    target = st.selectbox("Alege targetul", options=df_ml.columns, key="ml_target")
    default_features = [c for c in df_ml.columns if c != target]
    features = st.multiselect("Alege variabilele predictive", options=default_features, default=default_features, key="ml_features")

    dataML = df_ml[[target] + features].dropna(subset=[target])

    if 'genres' in features or target == 'genres':
        dataML = dataML.explode('genres')
    if 'top_cast' in features or target == 'top_cast':
        dataML = dataML.explode('top_cast')

    X = dataML[features]
    y = dataML[target]

    prob_type = st.selectbox("Alege:", ("regression", "classification"), key="ml_pred_select")

    numeric_cols_X = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_X = X.select_dtypes(include=['object']).columns.tolist()

    pipeline_numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    pipeline_categoric = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[("num", pipeline_numeric, numeric_cols_X),("cat", pipeline_categoric, cat_cols_X)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=y if (prob_type == "classification" and y.nunique() > 1) else None
    )

    if prob_type == "classification":
        catalog = {"Regresie Logistica": LogisticRegression(max_iter=1000), "Random Forest": RandomForestClassifier(random_state=42)}
        chosen_models = list(catalog.keys())
    else:
        catalog = {"Regresie Lineara": LinearRegression(), "Random Forest": RandomForestRegressor(random_state=42)}
        chosen_models = list(catalog.keys())

    if st.button("Start", type="primary"):
        results = []
        
        for name in chosen_models:
            pipe = Pipeline([("preprocess", preprocessor), ("model", catalog[name])])
            pipe.fit(X_train, y_train)
            
            y_pred = pipe.predict(X_test)
            row = {"Model": name}
            
            if prob_type == "classification":
                row["Accuracy"] = accuracy_score(y_test, y_pred)
                row["F1-Score (Weighted)"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            if prob_type == "regression":
                row["R2 Score"] = r2_score(y_test, y_pred)
                row["Mean Absolute Error"] = mean_absolute_error(y_test, y_pred)  
            results.append(row)
        st.dataframe(pd.DataFrame(results), use_container_width=True)


selected_module = sidebar_navigation()
if selected_module == "Vizualizare si manipulare date":
    manipulare_date()
if selected_module == "Analiza Partea 1 - Analiza Valori Numerice":
    analiza_date_partea_1()
if selected_module == "Analiza Partea 2 - Analiza Valori Categorice":
    analiza_date_partea_2()
if selected_module == "Analiza Partea 3 - Analiza Predictiva":
    machine_learning_partea_3()