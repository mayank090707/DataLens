import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# ---------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------
st.set_page_config(
    page_title="DataLens | Student Analytics",
    page_icon="",
    layout="wide"
)
sns.set_theme(style="whitegrid")

# ---------------------------------------
# 2. LOAD DATASET
# ---------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "cleaned_students.csv"


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    # Create performance categories
    def categorize_grade(grade):
        if grade >= 16:
            return "Excellent"
        elif grade >= 12:
            return "Good"
        elif grade >= 10:
            return "Average"
        else:
            return "Below Pass"

    data["Performance"] = data["G3"].apply(
        categorize_grade
    )

    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found! Please place cleaned_students.csv "
        "inside the data folder."
    )
    st.stop()

# ---------------------------------------
# 3. CUSTOM STYLING
# ---------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #F7F9FC;
    }

    h1 {
        color: #17365D;
    }

    .metric-card {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #E5EAF0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# 4. DASHBOARD HEADER
# ---------------------------------------

st.title("📊 DataLens")
st.subheader("Student Performance Analysis Dashboard")

st.markdown(
    "Explore student academic performance through "
    "interactive data analysis and visualizations."
)

st.divider()

# ---------------------------------------
# 5. SIDEBAR FILTERS
# ---------------------------------------

st.sidebar.title("Dashboard Filters")
st.sidebar.write("Customize your dataset view.")

selected_school = st.sidebar.multiselect(
    "Select School",
    options=sorted(df["school"].dropna().unique()),
    default=sorted(df["school"].dropna().unique())
)

selected_gender = st.sidebar.multiselect(
    "Select Gender",
    options=sorted(df["sex"].dropna().unique()),
    default=sorted(df["sex"].dropna().unique()),
    format_func=lambda x: "Female" if x == "F" else "Male"
)

selected_studytime = st.sidebar.multiselect(
    "Select Study Time",
    options=sorted(df["studytime"].dropna().unique()),
    default=sorted(df["studytime"].dropna().unique()),
    format_func=lambda x: {
        1: "Less than 2 hours",
        2: "2–5 hours",
        3: "5–10 hours",
        4: "More than 10 hours"
    }.get(x, str(x))
)

min_age = int(df["age"].min())
max_age = int(df["age"].max())

age_range = st.sidebar.slider(
    "Select Age Range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

# ---------------------------------------
# 6. FILTER DATA
# ---------------------------------------

filtered_df = df[
    (df["school"].isin(selected_school)) &
    (df["sex"].isin(selected_gender)) &
    (df["studytime"].isin(selected_studytime)) &
    (df["age"].between(age_range[0], age_range[1]))
].copy()

if filtered_df.empty:
    st.warning("No students match the selected filters.")
    st.stop()

# ---------------------------------------
# 7. KEY PERFORMANCE INDICATORS
# ---------------------------------------

total_students = len(filtered_df)

average_grade = filtered_df["G3"].mean()

passed_students = (filtered_df["G3"] >= 10).sum()

below_pass_students = (filtered_df["G3"] < 10).sum()

pass_percentage = (
    passed_students / total_students
) * 100

st.header("Academic Performance Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Students",
    total_students
)

col2.metric(
    "Average Final Grade",
    f"{average_grade:.2f}/20"
)

col3.metric(
    "Pass Percentage",
    f"{pass_percentage:.2f}%"
)

col4.metric(
    "Below-Pass Students",
    below_pass_students
)

st.divider()

# ---------------------------------------
# 8. VISUALIZATION 1: GRADE DISTRIBUTION
# ---------------------------------------

st.header("1. Final Grade Distribution")

fig1, ax1 = plt.subplots(figsize=(10, 5))

sns.histplot(
    data=filtered_df,
    x="G3",
    bins=21,
    color="royalblue",
    edgecolor="black",
    ax=ax1
)

ax1.set_title("Distribution of Final Grades")
ax1.set_xlabel("Final Grade")
ax1.set_ylabel("Number of Students")

st.pyplot(fig1)
plt.close(fig1)

st.info(
    "This histogram shows how students' final grades "
    "are distributed across the dataset."
)

# ---------------------------------------
# 9. VISUALIZATION 2: STUDY TIME
# ---------------------------------------

st.header("2. Study Time vs Student Performance")

fig2, ax2 = plt.subplots(figsize=(9, 5))

sns.boxplot(
    data=filtered_df,
    x="studytime",
    y="G3",
    color="skyblue",
    ax=ax2
)

ax2.set_title("Study Time vs Final Grades")
ax2.set_xlabel("Weekly Study Time Category")
ax2.set_ylabel("Final Grade")

st.pyplot(fig2)
plt.close(fig2)

# ---------------------------------------
# 10. VISUALIZATION 3: ABSENCES
# ---------------------------------------

st.header("3. Absences vs Final Grades")

fig3, ax3 = plt.subplots(figsize=(10, 5))

sns.regplot(
    data=filtered_df,
    x="absences",
    y="G3",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"},
    ax=ax3
)

ax3.set_title("Absences vs Final Grades")
ax3.set_xlabel("Number of Absences")
ax3.set_ylabel("Final Grade")

st.pyplot(fig3)
plt.close(fig3)

absence_corr = filtered_df["absences"].corr(
    filtered_df["G3"]
)

st.write(
    "Correlation between absences and final grades:",
    round(absence_corr, 3)
)

# ---------------------------------------
# 11. VISUALIZATION 4: CORRELATION HEATMAP
# ---------------------------------------

st.header("4. Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(
    include="number"
)

correlation_matrix = numeric_df.corr()

fig4, ax4 = plt.subplots(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    ax=ax4
)

ax4.set_title("Correlation Between Numerical Variables")

st.pyplot(fig4)
plt.close(fig4)

# ---------------------------------------
# 12. VISUALIZATION 5: GENDER AND SCHOOL
# ---------------------------------------

st.header("5. Performance by Gender and School")

fig5, ax5 = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=filtered_df,
    x="school",
    y="G3",
    hue="sex",
    palette="Set2",
    errorbar=None,
    ax=ax5
)

ax5.set_title("Average Final Grades by Gender and School")
ax5.set_xlabel("School")
ax5.set_ylabel("Average Final Grade")

st.pyplot(fig5)
plt.close(fig5)

# ---------------------------------------
# 13. STUDENT DATA EXPLORER
# ---------------------------------------

st.divider()

st.header("Student Data Explorer")

st.write(
    "Explore the student records based on your selected filters."
)

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------
# 14. DOWNLOAD DATASET
# ---------------------------------------

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇ Download Filtered Dataset",
    data=csv_data,
    file_name="datalens_filtered_students.csv",
    mime="text/csv"
)

# ---------------------------------------
# 15. FOOTER
# ---------------------------------------

st.divider()

st.caption(
    "DataLens | Student Performance Analysis "
    "| Built using Python, Pandas, Matplotlib, "
    "Seaborn and Streamlit"
)
