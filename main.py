import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
import os

# -----------------------------
# 1. Load Dataset
# -----------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Dataset not found at:\n{path}")

    df = pd.read_csv(path)
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    return df


# -----------------------------
# 2. Data Quality Check
# -----------------------------
def data_quality_check(df):
    print("\n--- Data Quality Check ---")
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    print("\nNumber of duplicate rows:")
    print(df.duplicated().sum())


# -----------------------------
# 3. Data Cleaning
# -----------------------------
def remove_duplicates(df):
    print("\n--- Removing Duplicate Rows ---")
    print("Rows before:", df.shape[0])
    df = df.drop_duplicates()
    print("Rows after:", df.shape[0])
    return df


# -----------------------------
# 4. Feature Engineering
# -----------------------------
def feature_engineering(df):
    print("\n--- Converting Date Columns ---")
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

    print("\n--- Calculating Hospital Stay Duration ---")
    df['Hospital Stay (Days)'] = (
        df['Discharge Date'] - df['Date of Admission']
    ).dt.days

    return df


# -----------------------------
# 5. Core Analysis
# -----------------------------
def hospital_stay_stats(df):
    print("\n--- Hospital Stay Statistics ---")
    print(df['Hospital Stay (Days)'].describe())


def admission_type_analysis(df):
    print("\n--- Hospital Stay by Admission Type ---")
    stay = df.groupby('Admission Type', observed=True)['Hospital Stay (Days)'].mean()
    print(stay)
    return stay


def age_analysis(df):
    print("\n--- Age vs Hospital Stay Analysis ---")
    corr = df['Age'].corr(df['Hospital Stay (Days)'])
    print("Correlation:", corr)


def medical_condition_analysis(df):
    print("\n--- Medical Condition vs Hospital Stay ---")
    print(
        df.groupby('Medical Condition')['Hospital Stay (Days)']
        .mean()
        .sort_values(ascending=False)
    )


def billing_analysis(df):
    print("\n--- Billing Amount vs Hospital Stay Analysis ---")
    corr = df['Billing Amount'].corr(df['Hospital Stay (Days)'])
    print("Correlation:", corr)


def insurance_analysis(df):
    print("\n--- Insurance Provider vs Hospital Stay ---")
    print(
        df.groupby('Insurance Provider')['Hospital Stay (Days)']
        .mean()
        .sort_values(ascending=False)
    )


# ======================================================
# =============== VISUALIZATION SECTION =================
# ======================================================

def plot_histogram_3d(df):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    data = df['Hospital Stay (Days)']
    hist, bins = np.histogram(data, bins='sturges')

    xs = (bins[:-1] + bins[1:]) / 2
    zs = np.zeros_like(xs)

    ax.bar(xs, hist, zs, zdir='y', alpha=0.85)

    ax.set_xlabel("Hospital Stay (Days)")
    ax.set_zlabel("Number of Patients")
    ax.set_title("3D Distribution of Hospital Stay")

    plt.show()


def plot_pie_chart_3d(df):
    admission_counts = df['Admission Type'].value_counts()
    labels = admission_counts.index
    sizes = admission_counts.values

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    theta = np.linspace(0, 2 * np.pi, len(sizes) + 1)

    for i in range(len(sizes)):
        x = [0, np.cos(theta[i]), np.cos(theta[i+1])]
        y = [0, np.sin(theta[i]), np.sin(theta[i+1])]
        z = [0, 0, 0]

        ax.plot_trisurf(x, y, z, alpha=0.9)

        # ✅ ADD LABEL
        angle = (theta[i] + theta[i+1]) / 2
        ax.text(
            0.7 * np.cos(angle),
            0.7 * np.sin(angle),
            0.02,
            f"{labels[i]} ({sizes[i]})",
            ha='center',
            fontsize=9,
            color='black'
        )

    ax.set_title("3D Admission Type Distribution")
    ax.set_axis_off()
    plt.show()


def plot_boxplot_3d(df):
    categories = df['Admission Type'].unique()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i, cat in enumerate(categories):
        data = df[df['Admission Type'] == cat]['Hospital Stay (Days)']
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        min_val, max_val = data.min(), data.max()

        ax.plot([i, i], [0, 0], [min_val, max_val])
        ax.bar([i], [q3 - q1], zs=[q1], zdir='z', alpha=0.7)
        ax.scatter([i], [0], [median], color='red')

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_xlabel("Admission Type")
    ax.set_zlabel("Hospital Stay (Days)")
    ax.set_title("3D Hospital Stay by Admission Type")

    plt.show()


def plot_grouped_line(df):
    grouped = (
        df.groupby([df['Date of Admission'].dt.to_period('M'), 'Admission Type'])
        ['Hospital Stay (Days)']
        .mean()
        .unstack()
    )

    plt.figure(figsize=(10, 5))
    for col in grouped.columns:
        plt.plot(grouped.index.astype(str), grouped[col], label=col)

    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Average Hospital Stay (Days)")
    plt.title("Hospital Stay Trend by Admission Type")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_disease_scatter_3d(df):
    disease_stay = df.groupby('Medical Condition')['Hospital Stay (Days)'].mean().reset_index()

    x = np.arange(len(disease_stay))
    y = disease_stay['Hospital Stay (Days)']
    z = np.zeros(len(x))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, s=80)
    ax.set_xticks(x)
    ax.set_xticklabels(disease_stay['Medical Condition'], rotation=30)
    ax.set_xlabel("Medical Condition")
    ax.set_ylabel("Avg Stay (Days)")
    ax.set_title("3D Disease-wise Hospital Stay")

    plt.show()


def plot_age_group_3d(df):
    df['Age Group'] = pd.cut(
        df['Age'],
        bins=[0, 18, 35, 50, 65, 100],
        labels=['0-18', '19-35', '36-50', '51-65', '65+']
    )

    age_group_stay = df.groupby('Age Group', observed=True)['Hospital Stay (Days)'].mean()
    x = np.arange(len(age_group_stay))
    y = age_group_stay.values
    z = np.zeros(len(x))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar(x, y, z, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(age_group_stay.index.astype(str))
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Avg Stay (Days)")
    ax.set_title("3D Average Hospital Stay by Age Group")

    plt.show()


def plot_age_vs_stay_trend_3d(df):
    age_trend = df.groupby('Age')['Hospital Stay (Days)'].mean()

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(age_trend.index, age_trend.values, np.zeros(len(age_trend)))

    ax.set_xlabel("Age")
    ax.set_ylabel("Avg Hospital Stay")
    ax.set_title("3D Age vs Hospital Stay Trend")

    plt.show()


# -----------------------------
# 6. Main Execution
# -----------------------------
def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR, "data", "healthcare_dataset.csv")

    df = load_data(file_path)
    data_quality_check(df)
    df = remove_duplicates(df)
    df = feature_engineering(df)

    hospital_stay_stats(df)
    admission_type_analysis(df)
    age_analysis(df)
    medical_condition_analysis(df)
    billing_analysis(df)
    insurance_analysis(df)

    plot_histogram_3d(df)
    plot_pie_chart_3d(df)
    plot_boxplot_3d(df)
    plot_grouped_line(df)
    plot_disease_scatter_3d(df)
    plot_age_group_3d(df)
    plot_age_vs_stay_trend_3d(df)


if __name__ == "__main__":
    main()
