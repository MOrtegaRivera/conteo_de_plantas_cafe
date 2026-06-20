import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from folium.plugins import MarkerCluster
from streamlit_folium import st_folium


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Conteo de plantas de café con YOLOv11",
    page_icon="☕",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* ======================================================
       FONDO GENERAL Y BARRA SUPERIOR
       ====================================================== */

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main,
    .block-container {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }

    /* Barra superior de Streamlit */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        box-shadow: none !important;
        border-bottom: 1px solid #E5E7EB !important;
    }

    header[data-testid="stHeader"] * {
        color: #1F2937 !important;
        fill: #1F2937 !important;
    }

    div[data-testid="stDecoration"],
    div[data-testid="stToolbar"] {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }

    .block-container {
        padding-top: 2rem !important;
    }

    /* ======================================================
       BARRA LATERAL Y FILTROS
       ====================================================== */

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background-color: #F7F7F2 !important;
        color: #1F2937 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #6B8E23 !important;
        color: #1F2937 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: #1F2937 !important;
    }

    div[data-baseweb="popover"] div {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }

    div[role="radiogroup"] label,
    div[role="radiogroup"] span {
        color: #1F2937 !important;
    }

    /* ======================================================
       TEXTOS Y COMPONENTES
       ====================================================== */

    h1, h2, h3, h4, h5, h6, p, label, span, div, li {
        color: #1F2937;
    }

    div[data-testid="stMetric"] {
        background-color: #F7F7F2 !important;
        padding: 12px !important;
        border-radius: 10px !important;
        border: 1px solid #DAD7CD !important;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div {
        color: #1F2937 !important;
    }

    /* ======================================================
       TABLAS
       ====================================================== */

    div[data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DAD7CD !important;
        border-radius: 8px !important;
    }

    /* Encabezado de tablas en color verde */
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] th {
        background-color: #2E7D32 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stDataFrame"] td {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# ============================================================
# FUENTE DE DATOS
# ============================================================

URL_DATOS = "https://raw.githubusercontent.com/MOrtegaRivera/conteo_de_plantas_cafe/refs/heads/main/conteo_plantas_cafe_coordenadas.csv"

# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================

@st.cache_data
def cargar_datos():
    """
    Carga el conjunto de datos de conteo automatizado de plantas
    de café y prepara variables derivadas para la aplicación.
    """

    df = pd.read_csv(URL_DATOS, encoding="utf-8-sig")

    # Limpieza básica de nombres de columnas
    df.columns = df.columns.str.strip()

    # Conversión de columnas numéricas
    columnas_numericas = [
        "latitud",
        "longitud",
        "altitud_m",
        "pendiente_pct",
        "area_ha",
        "plantas_campo",
        "plantas_detectadas",
        "precision",
        "recall",
        "f1_score",
        "mae_conteo"
    ]

    for columna in columnas_numericas:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

    # Eliminar registros sin coordenadas
    df = df.dropna(
        subset=["latitud", "longitud"]
    )

    # Variables derivadas
    df["diferencia_plantas"] = (
        df["plantas_campo"] - df["plantas_detectadas"]
    )

    df["tasa_deteccion_pct"] = (
        df["plantas_detectadas"] / df["plantas_campo"] * 100
    )

    df["lote_id"] = (
        df["finca"].astype(str)
        + " - "
        + df["lote"].astype(str)
    )

    return df


def color_region(region):
    """
    Asigna colores a los marcadores del mapa según la región cafetalera.
    """

    colores = {
        "Valle Central": "blue",
        "Valle Occidental": "green",
        "Los Santos": "orange"
    }

    return colores.get(region, "gray")


def color_f1(f1_score) -> str:
    """Return a color code based on the F1 score."""
    if f1_score >= 0.90:
        return "green"
    elif f1_score >= 0.85:
        return "orange"
    else:
        return "red"


def aplicar_estilo_grafico(fig, titulo_leyenda: str):
    """Aplica fondo claro y leyendas legibles a los gráficos de Plotly."""
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1F2937", size=13),
        title_font=dict(color="#1F2937", size=18),
        legend=dict(
            title=dict(
                text=titulo_leyenda,
                font=dict(color="#1F2937", size=13)
            ),
            font=dict(color="#1F2937", size=12),
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#DAD7CD",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=70, b=90),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        linecolor="#1F2937",
        tickfont=dict(color="#1F2937"),
        title_font=dict(color="#1F2937"),
        zerolinecolor="#E5E7EB"
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        linecolor="#1F2937",
        tickfont=dict(color="#1F2937"),
        title_font=dict(color="#1F2937"),
        zerolinecolor="#E5E7EB"
    )

    return fig

# ============================================================
# CARGA CONTROLADA
# ============================================================

try:
    datos = cargar_datos()

except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
    st.error("No fue posible cargar los datos.")
    st.exception(error)
    st.stop()


# ============================================================
# ENCABEZADO
# ============================================================

st.title("☕ Conteo automatizado de plantas de café con YOLOv11")

st.markdown(
    """
    Esta aplicación web interactiva presenta los resultados de la
    **Tarea 3**, orientada al análisis de datos para el conteo
    automatizado de plantas de café mediante imágenes multiespectrales
    captadas con drones y el uso del modelo **YOLOv11**.

    El conjunto de datos contiene información de fincas y lotes
    cafetaleros de Costa Rica, distribuidos en distintas regiones
    productoras. Para cada lote se incluyen variables como altitud,
    pendiente, tipo de sombra, área, plantas contabilizadas en campo,
    plantas detectadas automáticamente y métricas de desempeño del modelo:
    precisión, recall, F1-score y MAE.
    """
)

st.info(
    """
    Utilice los filtros de la barra lateral para actualizar la tabla,
    los gráficos y el mapa de forma interactiva.
    """
)

# ============================================================
# FILTROS INTERACTIVOS
# ============================================================

st.sidebar.header("Filtros")

# Filtro por región
regiones = sorted(
    datos["region"].dropna().unique()
)

opciones_region = ["Todas"] + regiones

region_seleccionada = st.sidebar.selectbox(
    "Región cafetalera",
    options=opciones_region
)

if region_seleccionada == "Todas":
    datos_filtrados = datos.copy()
else:
    datos_filtrados = datos[
        datos["region"] == region_seleccionada
    ].copy()


# Filtro por finca
fincas = sorted(
    datos_filtrados["finca"].dropna().unique()
)

opciones_finca = ["Todas"] + fincas

finca_seleccionada = st.sidebar.selectbox(
    "Finca",
    options=opciones_finca
)

if finca_seleccionada != "Todas":
    datos_filtrados = datos_filtrados[
        datos_filtrados["finca"] == finca_seleccionada
    ].copy()


# Filtro por tipo de sombra
sombras = sorted(
    datos_filtrados["tipo_sombra"].dropna().unique()
)

opciones_sombra = ["Todas"] + sombras

sombra_seleccionada = st.sidebar.selectbox(
    "Tipo de sombra",
    options=opciones_sombra
)

if sombra_seleccionada != "Todas":
    datos_filtrados = datos_filtrados[
        datos_filtrados["tipo_sombra"] == sombra_seleccionada
    ].copy()


if datos_filtrados.empty:
    st.warning(
        "No hay datos disponibles para los filtros seleccionados."
    )
    st.stop()


st.sidebar.markdown("---")
st.sidebar.write(
    f"Registros seleccionados: **{len(datos_filtrados)}**"
)

# ============================================================
# INDICADORES GENERALES
# ============================================================

total_fincas = datos_filtrados["finca"].nunique()
total_lotes = len(datos_filtrados)
total_campo = datos_filtrados["plantas_campo"].sum()
total_detectadas = datos_filtrados["plantas_detectadas"].sum()
f1_promedio = datos_filtrados["f1_score"].mean()
mae_promedio = datos_filtrados["mae_conteo"].mean()

tasa_deteccion = (
    total_detectadas / total_campo * 100
    if total_campo > 0
    else 0
)

st.subheader("Indicadores generales")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Fincas",
    f"{total_fincas}"
)

col2.metric(
    "Lotes",
    f"{total_lotes}"
)

col3.metric(
    "Plantas en campo",
    f"{total_campo:,.0f}"
)

col4.metric(
    "Plantas detectadas",
    f"{total_detectadas:,.0f}"
)

col5.metric(
    "Detección",
    f"{tasa_deteccion:.1f} %"
)

col6, col7 = st.columns(2)

col6.metric(
    "F1-score promedio",
    f"{f1_promedio:.3f}"
)

col7.metric(
    "MAE promedio",
    f"{mae_promedio:.1f} plantas"
)

st.divider()


# ============================================================
# TABLA CON PANDAS
# ============================================================

st.header("1. Tabla de desempeño del modelo por finca")

st.markdown(
    """
    La siguiente tabla corresponde a una adaptación de la tabla generada
    en la Tarea 3. Resume el desempeño del modelo por finca, incluyendo
    plantas contabilizadas en campo, plantas detectadas automáticamente,
    F1-score promedio, MAE promedio y tasa de detección.
    """
)

tabla_finca = (
    datos_filtrados
    .groupby("finca", as_index=False)
    .agg(
        region=("region", "first"),
        tipo_sombra=(
            "tipo_sombra",
            lambda x: ", ".join(sorted(x.unique()))
        ),
        altitud_prom=("altitud_m", "mean"),
        pendiente_prom=("pendiente_pct", "mean"),
        plantas_campo=("plantas_campo", "sum"),
        plantas_detect=("plantas_detectadas", "sum"),
        f1_prom=("f1_score", "mean"),
        mae_prom=("mae_conteo", "mean")
    )
)

tabla_finca["tasa_deteccion_pct"] = (
    tabla_finca["plantas_detect"]
    / tabla_finca["plantas_campo"]
    * 100
)

tabla_finca["diferencia_plantas"] = (
    tabla_finca["plantas_campo"]
    - tabla_finca["plantas_detect"]
)

tabla_presentacion = tabla_finca.rename(
    columns={
        "finca": "Finca",
        "region": "Región",
        "tipo_sombra": "Tipo de sombra",
        "altitud_prom": "Altitud promedio (m)",
        "pendiente_prom": "Pendiente promedio (%)",
        "plantas_campo": "Plantas en campo",
        "plantas_detect": "Plantas detectadas",
        "diferencia_plantas": "Diferencia",
        "f1_prom": "F1-score promedio",
        "mae_prom": "MAE promedio",
        "tasa_deteccion_pct": "Tasa de detección (%)"
    }
)

tabla_estilizada = (
    tabla_presentacion
    .style
    .set_properties(
        **{
            "background-color": "#FFFFFF",
            "color": "#1F2937",
            "border-color": "#DAD7CD"
        }
    )
    .set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#2E7D32"),
                    ("color", "#FFFFFF"),
                    ("font-weight", "bold"),
                    ("border-color", "#DAD7CD")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#FFFFFF"),
                    ("color", "#1F2937"),
                    ("border-color", "#DAD7CD")
                ]
            }
        ]
    )
)

st.dataframe(
    tabla_estilizada,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Altitud promedio (m)": st.column_config.NumberColumn(
            format="%.0f"
        ),
        "Pendiente promedio (%)": st.column_config.NumberColumn(
            format="%.1f"
        ),
        "F1-score promedio": st.column_config.NumberColumn(
            format="%.3f"
        ),
        "MAE promedio": st.column_config.NumberColumn(
            format="%.1f"
        ),
        "Tasa de detección (%)": st.column_config.NumberColumn(
            format="%.1f %%"
        )
    }
)

st.divider()


# ============================================================
# GRÁFICOS
# ============================================================

st.header("2. Gráficos estadísticos")

st.markdown(
    """
    Los gráficos permiten comparar el conteo manual realizado en campo
    con la detección automática generada por el modelo, así como observar
    la relación entre la pendiente del terreno, el tipo de sombra y el
    error absoluto medio del conteo.
    """
)

opcion_grafico = st.radio(
    "Seleccione el gráfico que desea visualizar:",
    [
        "Conteo manual vs detección automática",
        "Pendiente vs MAE según tipo de sombra"
    ],
    horizontal=True
)


if opcion_grafico == "Conteo manual vs detección automática":

    df_largo = pd.melt(
        datos_filtrados,
        id_vars=[
            "finca",
            "lote",
            "region",
            "lote_id"
        ],
        value_vars=[
            "plantas_campo",
            "plantas_detectadas"
        ],
        var_name="metodo",
        value_name="cantidad"
    )

    df_largo["metodo"] = df_largo["metodo"].map(
        {
            "plantas_campo": "Conteo manual en campo",
            "plantas_detectadas": "Detectadas por el modelo"
        }
    )

    fig = px.bar(
        df_largo,
        x="lote_id",
        y="cantidad",
        color="metodo",
        barmode="group",
        color_discrete_map={
        "Conteo manual en campo": "#2E7D32",
        "Detectadas por el modelo": "#F9C74F"
    },
        title="Conteo manual en campo vs detección automática por lote",
        labels={
            "lote_id": "Lote",
            "cantidad": "Número de plantas",
            "metodo": "Método"
        },
        hover_data={
            "region": True
        }
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title="Finca y lote",
        yaxis_title="Número de plantas"
    )

    aplicar_estilo_grafico(fig, "Método")
    
    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        """
        Este gráfico permite identificar en cuáles lotes existe una mayor
        diferencia entre el conteo manual y el conteo generado por YOLOv11.
        """
    )


else:

    fig = px.scatter(
        datos_filtrados,
        x="pendiente_pct",
        y="mae_conteo",
        color="tipo_sombra",
        color_discrete_map={
            "Baja": "#27AE60",
            "Media": "#F39C12",
            "Alta": "#E74C3C"
        },
        title="Pendiente del terreno vs MAE del modelo según tipo de sombra",
        labels={
            "pendiente_pct": "Pendiente del terreno (%)",
            "mae_conteo": "Error absoluto medio - MAE (plantas)",
            "tipo_sombra": "Tipo de sombra"
        },
        hover_data={
            "finca": True,
            "lote": True,
            "region": True,
            "altitud_m": True,
            "f1_score": True
        },
        text="lote"
    )

    fig.update_traces(
        textposition="top center",
        textfont_size=9
    )

    if len(datos_filtrados) >= 2:
        m, b = np.polyfit(
            datos_filtrados["pendiente_pct"],
            datos_filtrados["mae_conteo"],
            1
        )

        xr = np.linspace(
            datos_filtrados["pendiente_pct"].min() - 1,
            datos_filtrados["pendiente_pct"].max() + 1,
            100
        )

        fig.add_trace(
            go.Scatter(
                x=xr,
                y=m * xr + b,
                mode="lines",
                name="Tendencia lineal",
                line=dict(
                    color="#6F4E37",
                    width=3,
                    dash="dash"
                )
            )
        )

    fig.update_layout(
        xaxis_title="Pendiente del terreno (%)",
        yaxis_title="MAE de conteo (plantas)"
    )

    aplicar_estilo_grafico(fig, "Tipo de sombra")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        """
        Este gráfico permite analizar si los lotes con mayor pendiente o
        mayor sombra presentan errores más altos en el conteo automatizado.
        """
    )

st.divider()


# ============================================================
# MAPA INTERACTIVO
# ============================================================

st.header("3. Mapa interactivo de localización de lotes")

st.markdown(
    """
    El mapa muestra la ubicación geográfica de los lotes evaluados y
    clasifica cada marcador de acuerdo con el **F1-score** obtenido por el
    modelo YOLOv11.

    Los marcadores verdes representan lotes con **F1-score ≥ 0.90**,
    los marcadores naranjas representan lotes con **0.85 ≤ F1-score < 0.90**
    y los marcadores rojos representan lotes con **F1-score < 0.85**.
    Al hacer clic sobre cada marcador se despliega información detallada
    del lote y de las métricas del modelo.
    """
)

latitud_centro = datos_filtrados["latitud"].mean()
longitud_centro = datos_filtrados["longitud"].mean()

mapa = folium.Map(
    location=[
        latitud_centro,
        longitud_centro
    ],
    zoom_start=9,
    tiles="CartoDB positron",
    control_scale=True
)

folium.TileLayer(
    "OpenStreetMap",
    name="OpenStreetMap"
).add_to(mapa)

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri World Imagery",
    name="Imagen satelital"
).add_to(mapa)

cluster = MarkerCluster(
    name="Lotes de café"
).add_to(mapa)


for _, lote in datos_filtrados.iterrows():

    popup_html = f"""
    <div style='font-family:Arial; font-size:13px; min-width:230px;'>

        <b>{lote['finca']} - {lote['lote']}</b><br>

        <hr style='margin:4px 0'>

        <b>Región:</b> {lote['region']}<br>
        <b>Distrito:</b> {lote['distrito']}<br>
        <b>Altitud:</b> {lote['altitud_m']:.0f} m<br>
        <b>Pendiente:</b> {lote['pendiente_pct']:.1f} %<br>
        <b>Sombra:</b> {lote['tipo_sombra']}<br>
        <b>Área:</b> {lote['area_ha']:.2f} ha<br>

        <hr style='margin:4px 0'>

        <b>Plantas campo:</b> {lote['plantas_campo']:,.0f}<br>
        <b>Detectadas:</b> {lote['plantas_detectadas']:,.0f}<br>
        <b>Diferencia:</b> {lote['diferencia_plantas']:,.0f}<br>
        <b>Tasa detección:</b> {lote['tasa_deteccion_pct']:.1f} %<br>

        <hr style='margin:4px 0'>

        <b>Precisión:</b> {lote['precision']:.3f}<br>
        <b>Recall:</b> {lote['recall']:.3f}<br>
        <b>F1-score:</b> {lote['f1_score']:.3f}<br>
        <b>MAE:</b> {lote['mae_conteo']:.0f} plantas

    </div>
    """

    folium.Marker(
        location=[
            lote["latitud"],
            lote["longitud"]
        ],
        popup=folium.Popup(
            popup_html,
            max_width=300
        ),
        tooltip=folium.Tooltip(
            f"{lote['lote_id']} | F1 = {lote['f1_score']:.2f}"
        ),
        icon=folium.Icon(
            color=color_f1(lote["f1_score"]),
            icon="leaf",
            prefix="fa"
        )
    ).add_to(cluster)

leyenda_f1 = """
<div style="
    position: fixed;
    bottom: 70px;
    left: 70px;
    width: 285px;
    z-index: 9999;
    background-color: rgba(20, 20, 20, 0.92);
    color: white;
    border: 2px solid #F4D35E;
    border-radius: 8px;
    padding: 12px;
    font-size: 13px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.5);
">
    <b style="font-size:14px;">Simbología por F1-score</b><br><br>

    <i class="fa fa-leaf" style="color:#2ECC71; font-size:16px;"></i>
    &nbsp;Alto desempeño: F1 ≥ 0.90<br>

    <i class="fa fa-leaf" style="color:#F39C12; font-size:16px;"></i>
    &nbsp;Desempeño medio: 0.85 ≤ F1 &lt; 0.90<br>

    <i class="fa fa-leaf" style="color:#E74C3C; font-size:16px;"></i>
    &nbsp;Bajo desempeño: F1 &lt; 0.85
</div>
"""

mapa.get_root().html.add_child(
    folium.Element(leyenda_f1)
)

folium.LayerControl(
    collapsed=False
).add_to(mapa)

st_folium(
    mapa,
    use_container_width=True,
    height=600,
    returned_objects=[]
)

st.divider()


# ============================================================
# DATOS DETALLADOS
# ============================================================

with st.expander("Ver registros detallados"):

    columnas_detalle = [
        "finca",
        "lote",
        "region",
        "distrito",
        "altitud_m",
        "pendiente_pct",
        "tipo_sombra",
        "area_ha",
        "plantas_campo",
        "plantas_detectadas",
        "diferencia_plantas",
        "tasa_deteccion_pct",
        "precision",
        "recall",
        "f1_score",
        "mae_conteo",
        "latitud",
        "longitud"
    ]

    tabla_detalle_estilizada = (
        datos_filtrados[columnas_detalle]
        .style
        .set_properties(
            **{
                "background-color": "#FFFFFF",
                "color": "#1F2937",
                "border-color": "#DAD7CD"
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#2E7D32"),
                        ("color", "#FFFFFF"),
                        ("font-weight", "bold"),
                        ("border-color", "#DAD7CD")
                    ]
                },
                {
                    "selector": "td",
                    "props": [
                        ("background-color", "#FFFFFF"),
                        ("color", "#1F2937"),
                        ("border-color", "#DAD7CD")
                    ]
                }
            ]
        )
    )

    st.dataframe(
        tabla_detalle_estilizada,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# CONCLUSIONES
# ============================================================

st.header("4. Conclusiones generales")

st.markdown(
    """
    A partir de los resultados obtenidos, se identifican los siguientes
    hallazgos principales:

    - **Desempeño general del modelo:** YOLOv11 presentó un desempeño favorable
      para el conteo automatizado de plantas de café. En total se registraron
      **33 600 plantas contabilizadas en campo** y **32 475 plantas detectadas
      automáticamente**, lo que equivale a una **tasa general de detección de
      96.7 %**. La diferencia total fue de **1 125 plantas**, evidenciando una
      ligera subestimación del conteo real.

    - **Fincas con mejor desempeño:** Las mejores tasas de detección se
      observaron en **Finca 5**, con **98.1 %**, y **Finca 2**, con **97.9 %**.
      Estas fincas también registraron errores promedio bajos, con un **MAE de
      57.5** y **65.0 plantas**, respectivamente, lo que indica mayor cercanía
      entre el conteo manual y el conteo automatizado.

    - **Fincas con mayores diferencias:** Las diferencias más altas entre
      plantas contabilizadas y plantas detectadas se presentaron en **Finca 3**
      con **240 plantas**, **Finca 4** con **230 plantas** y **Finca 6** con
      **225 plantas**. Estos casos señalan sitios donde el modelo puede requerir
      mayor validación o ajuste.

    - **Influencia del tipo de sombra:** La sombra tuvo una relación clara con
      el desempeño del modelo. Los lotes con **sombra baja** alcanzaron una
      **tasa de detección de 98.8 %**, un **F1-score promedio de 0.923** y un
      **MAE promedio de 43.3 plantas**. En cambio, los lotes con **sombra alta**
      registraron el desempeño más bajo, con **93.0 % de detección**,
      **F1-score promedio de 0.823** y **MAE promedio de 153.8 plantas**.

    - **Relación entre pendiente y error:** Los mayores errores se concentran
      en lotes con pendientes fuertes. El caso más crítico corresponde a
      **Finca 4 - Lote 2**, en la región de **Los Santos**, con **35 % de
      pendiente**, **sombra alta**, **92.0 % de detección**, **F1-score de 0.81**
      y **MAE de 180 plantas**.

    - **Interpretación espacial:** El mapa muestra que el desempeño no depende
      únicamente de la región cafetalera, sino de las condiciones específicas
      de cada lote. En una misma región pueden coexistir lotes con alto y medio
      desempeño, por lo que el análisis espacial ayuda a priorizar zonas que
      requieren revisión en campo.

    - **Conclusión final:** El conteo automatizado de plantas de café con
      **YOLOv11** muestra alto potencial como herramienta de apoyo para el
      monitoreo agrícola. Sin embargo, los resultados indican que el desempeño
      disminuye principalmente en sitios con **sombra alta** y **pendientes
      fuertes**, por lo que futuras mejoras deberían incorporar más muestras
      de entrenamiento bajo estas condiciones.
    """
)