import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import json
from PIL import Image
import os
import matplotlib.pyplot as plt
import squarify


MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

st.title("Znajdź przyjaciół.")

with st.sidebar:
    st.header("Powiedz nam coś o sobie.")
    st.markdown("Pomożemy znaleźć osoby do Ciebie")
    age= st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '55-64', '>64'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])
    
    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()
predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]


# st.write("Wybrane dane:")
# st.dataframe(person_df, hide_index=True)

# st.write("Przykładowe osoby z bazy:")
# st.dataframe(all_df.sample(10), hide_index=True) 


predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
# obrazek
    # Nazwa pliku obrazka
image_file = f"{predicted_cluster_id}.jpeg"

    # Sprawdzenie, czy plik istnieje
if os.path.exists(image_file):
    # Wyświetlanie obrazka w Streamlit
    image = Image.open(image_file)
    new_width = int(image.width * 0.5)  # 50% oryginalnej szerokości
    new_height = int(image.height * 0.5)  # 50% oryginalnej wysokości
    st.image(image, width=new_width, use_container_width=False)
else:
    st.write(f"Plik {image_file} nie istnieje.")
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]

st.header("Osoby z grupy")
st.markdown("**Rozkład wieku w grupie**")
# Wykres drzewo
# Ustawienia wykresu
plt.figure(figsize=(10, 6))  # Rozmiar wykresu
# Dane
age_summary = same_cluster_df['age'].value_counts().reset_index()
age_summary.columns = ['age', 'count']
age_summary=age_summary[age_summary['count'] > 0]
# st.dataframe(age_summary)

# Tworzenie mapy drzewa
fig, ax = plt.subplots(facecolor='black')
colors = plt.cm.Blues(age_summary['count'] / max(age_summary['count']))
squarify.plot(sizes=age_summary['count'], label=age_summary['age'], alpha=.8, color=colors)

# Tytuł wykresu
# plt.title('Mapa drzewa (Treemap)', fontsize=18)
plt.axis('off')  # Wyłącza osie

# Wyświetlenie wykresu
st.pyplot(fig)

# fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
# fig.update_layout(
#     title="Rozkład wieku w grupie",
#     xaxis_title="Wiek",
#     yaxis_title="Liczba osób",
# )
# st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)



