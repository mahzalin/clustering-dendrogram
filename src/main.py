import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc


def fetch_dataset(file_path):
    """Read CSV file and return DataFrame."""
    return pd.read_csv(file_path)


def show_dendrogram(clusters):
    """Plot dendrogram from linkage matrix."""
    plt.figure(figsize=(12.2, 6))
    shc.dendrogram(Z=clusters)
    plt.xticks(rotation=45)
    plt.title('Course Grade Dendrogram')
    plt.xlabel('Row Number')
    plt.ylabel('Course Grade')
    plt.tight_layout()
    plt.show()


def show_histogram(data, field_name):
    """Plot histogram of a specific field."""
    data[field_name].hist()
    plt.title(f'Histogram of {field_name}')
    plt.xlabel(field_name)
    plt.ylabel('Frequency')
    plt.show()


def validate_sex_column(data):
    """Check if 'sex' column has exactly two unique values."""
    sex_array = data['sex'].unique()
    if len(sex_array) != 2:
        raise ValueError("The dataset has invalid or unexpected 'sex' values.")
    print("Sex column distribution:\n", data['sex'].value_counts(normalize=True))


def group_course_grades(data):
    """Create grade group column based on course_grade."""
    bins = [20, 40, 60, 80, 100]
    labels = ['(20-40]', '(40-60]', '(60-80]', '(80-100]']
    data['Course Grade Group'] = pd.cut(data['course_grade'], bins=bins, labels=labels)
    print("\nCount per grade group:\n", data['Course Grade Group'].value_counts())
    return data


def preprocess_data(data):
    """Drop unnecessary columns and apply one-hot encoding."""
    columns_to_drop = ['semester', 'exam1', 'exam2', 'exam3']
    data = data.drop(columns=columns_to_drop)
    data_encoded = pd.get_dummies(data)
    return data_encoded


def main():
    data = fetch_dataset('exam_grades.csv')

    # Basic EDA
    show_histogram(data, 'course_grade')
    print("\nDataset Summary:\n", data.describe().transpose())
    print("\nDataset Info:")
    data.info()

    # Validate and group
    validate_sex_column(data)
    data = group_course_grades(data)

    # Prepare for clustering
    processed_data = preprocess_data(data)
    features = processed_data.iloc[:, :2]  # Select first 2 columns for simplicity
    clusters = shc.linkage(features, method='ward', metric='euclidean')

    show_dendrogram(clusters)


if __name__ == '__main__':
    main()
