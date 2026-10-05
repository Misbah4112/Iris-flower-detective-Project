#import libraries
import os #allows the program to interact with os and work on files,folders,environment variables
import pandas as pd #Load, organize, and clean data in tables with rows and columns.
import matplotlib.pyplot as plt #used for data visualization
import streamlit as st #used to create web apps for data science and machine learning
from sklearn.ensemble import RandomForestClassifier #used to create a random forest model for classification tasks
from sklearn.model_selection import train_test_split #used to split the dataset into training and testing sets

from sklearn.metrics import (accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report) #used to evaluate the performance of the model


#Streamlit page settings
st.set_page_config(
    page_title = "Flower Detective",
    page_icon = "🤗",
    layout = "wide"

)

#css design
st.markdown("""
<style>
.stApp {
    background-color: #0e1113;
    color: #eef0f2;
}

.block-container {
    max-width: 1000px;
    padding-top: 3rem;
}

.iris-card {
    background: linear-gradient(120deg, #15191c, #101215);
    border: 1px solid #25292d;
    border-radius: 14px;
    padding: 24px;
    font-family: sans-serif;
}

.iris-card .eyebrow {
    color: #a5a9ae;
    font-size: 13px;
    margin: 0 0 20px;
}

.iris-card h1 {
    color: #eef0f2;
    font-size: 34px;
    font-weight: 750;
    line-height: 1.2;
    margin: 0 0 18px;
    padding: 0;
}

.iris-card h1 span {
    color: #f58a45;
}

.iris-card .description {
    color: #a5a9ae;
    font-size: 14px;
    line-height: 1.6;
    margin: 0;
}

@media (max-width: 600px) {
    .iris-card {
        padding: 20px;
    }

    .iris-card h1 {
        font-size: 28px;
    }
}
</style>

<div class="iris-card">
    <p class="eyebrow">Academy 360 • Machine Learning Foundations</p>
    <h1>Iris <span>Flower Detective</span></h1>
    <p class="description">
        Change the four flower measurements and let a Random Forest
        classifier predict the Iris species.
    </p>
</div>
""", unsafe_allow_html=True)
#project file path
BASE_DIR = os.path.dirname(
  os.path.abspath(__file__)
 )

#__file__ → the current Python file, such as app.py.
#os.path.abspath(__file__) → gets its full path,
#os.path.dirname(...) → removes the filename

DATA_PATH = os.path.join(BASE_DIR, "iris.csv")
#os.path.join() combines the folder path with the filename iris.csv


#load the dataset
@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    dataframe = pd.read_csv(DATA_PATH)

    return dataframe


try:

    df = load_data()
except FileNotFoundError as error:
    st.error(
        str(error)
    )

feature_columns = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm"
]


#Define features and target
X =df[feature_columns]
y = df["species"]

X_train,X_test,y_train,y_test = train_test_split(
X,
y,
test_size=0.2,
random_state = 42,
stratify =y


)

##create and train mode;
@st.cache_resource
def train_model():
    classifier = RandomForestClassifier(
        n_estimators = 100,
        random_state = 42
    )

    classifier.fit(X_train,y_train)

    return classifier

model = train_model()

##test model on unseen flowers

test_predictions = model.predict(X_test)

accuracy = accuracy_score(y_test,test_predictions)

cm = confusion_matrix(y_test,test_predictions,
labels = model.classes_)

cn_df = pd.DataFrame(
    cm,
    index = [f"Actual {name.title()}" for name in model.classes_],
    columns = [f"Predicted {name.title()}" for name in model.classes_]
)

## app header

##top summary metrics
metric_1,metric_2,metric_3,metric_4 = st.columns(4)

metric_1.metric(
    "Flowers",
    len(df)
    )


metric_2.metric(
    "Input Features",
    len(feature_columns)
    )

metric_3.metric(
    "Classes",
    y.nunique() 
      )


metric_4.metric(
    "Test Accuracy",
    f"{accuracy:.2f}"
    )

st.divider()

prediction_tab, performance_tab,data_tab,learn_tab = st.tabs(
    [
        "🌸 Predict a Flower",
        "📈 Model Performance",
        "💿 Dataset Exploration",
        "🧠 Learn the Project"

    ]
)

## New Flower Prediction
with prediction_tab:
    st.subheader("Flower Measurements")
    st.write("Move the sliders, then press **Predict Species**")

    with st.form("prediction_form"):
        left_col, right_col = st.columns(2)

        with left_col:
            sepal_length = st.slider(
                "Sepal Length (cm)",
                min_value=float(df["sepal_length_cm"].min()),
                max_value=float(df["sepal_length_cm"].max()),
                value=float(df["sepal_length_cm"].mean()),
                step=0.1
            )

            sepal_width = st.slider(
                "Sepal Width (cm)",
                min_value=float(df["sepal_width_cm"].min()),
                max_value=float(df["sepal_width_cm"].max()),
                value=float(df["sepal_width_cm"].mean()),
                step=0.1
            )

        with right_col:
            petal_length = st.slider(
                "Petal Length (cm)",
                min_value=float(df["petal_length_cm"].min()),
                max_value=float(df["petal_length_cm"].max()),
                value=float(df["petal_length_cm"].mean()),
                step=0.1
            )

            petal_width = st.slider(
                "Petal Width (cm)",
                min_value=float(df["petal_width_cm"].min()),
                max_value=float(df["petal_width_cm"].max()),
                value=float(df["petal_width_cm"].mean()),
                step=0.1
            )

        predict_button = st.form_submit_button(
            "Predict Species",
            type="primary",
            use_container_width=True
        )

    input_flower = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=feature_columns
    )

    with st.expander("See Input Flower Measurements"):
        st.dataframe(input_flower, use_container_width=True)

    if predict_button:
        prediction = model.predict(input_flower)[0]
        prediction_proba = model.predict_proba(input_flower)[0]

        st.success(f"The predicted species is: **{prediction.title()}**")

        highest_probability = prediction_proba.max()
        st.metric("Highest Model Probability: ", f"{highest_probability:.2f}")

        probability_df = pd.DataFrame(
            {"Species": [name.title() for name in model.classes_],
             "Probability": prediction_proba}
        ).set_index("Species")

        st.write("MOdel confidence by species:")
        st.bar_chart(probability_df)

        st.caption(
            "The probability values represent the model's confidence in its prediction. it should not be treated as absolute certainty."
        )


with performance_tab:
    st.subheader("How well does the model perform on unseen flowers?")
    correct_predictions = int((test_predictions == y_test).sum())

    st.write(
        f"The model correctly predicted **{correct_predictions}** out of **{len(y_test)}** unseen flowers in the test set."
    )
    st.write("Confusion Matrix:")
    st.write("Rows show actual species, while columns show predicted species.")
    st.dataframe(cn_df, use_container_width=True)
    st.write("Precision, Recall, and F1-Score:")
    report = classification_report(y_test, test_predictions, output_dict=True,zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df, use_container_width=True)


    st.write("Feature Importance:")
    st.write("Feature importance shows which measurements the trained random forest model considers most important when making predictions.")

    importance_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }
    ).sort_values(by="Importance", ascending=False
    )

    st.dataframe(importance_df.round(2), use_container_width=True,hide_index=True)
    st.bar_chart(importance_df.set_index("Feature"))

with data_tab:
    st.subheader("Dataset Exploration")
    st.write("The dataset contains measurements of iris flowers, including sepal length, sepal width, petal length, petal width, and species.")

    species_filter = st.selectbox(
        "Choose a species to view",
        ["All Species"]+list(df["species"].unique())
    
    )

    if species_filter == "All Species":
                filtered_df = df
    else:
        filtered_df = df[df["species"] == species_filter.lower()]
    show_table = st.checkbox("Show Data Table", value=True)
    if show_table:
        st.dataframe(filtered_df, use_container_width=True)

    st.write("Species Counts:")
    species_counts = filtered_df["species"].value_counts()
    st.bar_chart(species_counts)

    st.write('Numerical Summary Statistics:')
    st.dataframe(filtered_df.describe().round(2), use_container_width=True)

with learn_tab:
    st.subheader("Complete Machine Learning Workflow")

    st.write("**1. CSV Data** – We start with 150 labelled Iris flowers.")
    st.write("**2. Features + Target** – Four measurements are X, species is y.")
    st.write("**3. Train/Test Split** – 80% training and 20% testing.")
    st.write("**4. Random Forest** – 100 decision trees learn from training data.")
    st.write("**5. Evaluation** – We inspect accuracy and the model's mistakes.")
    st.write("**6. Streamlit Input** – The user chooses four measurements.")
    st.write("**7. Prediction** – The model predicts one of three species.")
    st.write("**8. Deployment** – The project can be uploaded and shared online.")

  
st.divider()
st.caption("Learn , Practice and Grow")