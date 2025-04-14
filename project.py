# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For advanced visualizations
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.neighbors import KNeighborsClassifier  # For KNN classification
from time import sleep
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import silhouette_score

# Open file dialog to choose any .xlsx or .csv file
root = tk.Tk()
root.withdraw()  # Hide the root window
file_path = filedialog.askopenfilename(
    title="Select a file",
    filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")]
)

# Load the file based on its extension
if file_path.endswith('.xlsx'):
    sorted_db = pd.read_excel(file_path, 0)
elif file_path.endswith('.csv'):
    sorted_db = pd.read_csv(file_path)
else:
    raise ValueError("Selected file is neither a valid Excel nor CSV file.")

# Handle missing values (if any)
sorted_db.fillna(sorted_db.mean(numeric_only=True), inplace=True)

# Ask the user for feature columns
print("\n\n\n\n\n")
numerical_columns = sorted_db.select_dtypes(include=[np.number]).columns # Filter columns with numerical values only
print("Available columns (Numerical only):", list(numerical_columns))
feature_columns = []
while len(feature_columns) < 2:
    col = input(f"Select feature column {len(feature_columns) + 1}: ")
    if col in numerical_columns:
        feature_columns.append(col)
    else:
        print("Invalid selection. Ensure the column exists and contains numeric values.\n")
print("\n")

# Determine if we can use either KNN
binary_columns = [col for col in sorted_db.columns if sorted_db[col].nunique() == 2] # Filter columns with exactly two unique values
tech = 1 if binary_columns else 0

# Ask if we use either KNN or K-Means if the choice exists
if tech:
    while True:
        choice = input("Do you wanna use either KNN or K-Means ?\n")
        if choice != "KNN" and choice != "K-Means":
            print("Enter one of the 2 propositions (KNN/K-Means).\n\n")
            continue
        elif choice == "KNN":
            tech = 1
        else:
            tech = 0
        print("\n\n")
        break


# Test if we should use either KNN or K-Means
if tech:
    #KNN
    # Ask the user for the target column
    print("Available columns (Binary only):", list(binary_columns))
    while True:
        target_column = input("Select the target column: ")
        if target_column in binary_columns:
            unique_values = sorted_db[target_column].unique()
            target_mapping = {unique_values[0]: 0, unique_values[1]: 1}
            sorted_db['Target_Binary'] = sorted_db[target_column].map(target_mapping)
            break
        else:
            print("Invalid selection. Ensure the column exists and contains exactly two unique values.\n")
    print("\n\n")


    # Prepare data for splitting and training
    X = sorted_db[feature_columns].values
    y = sorted_db['Target_Binary'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Plot the original data
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=sorted_db[feature_columns[0]],
        y=sorted_db[feature_columns[1]],
        hue=sorted_db['Target_Binary'],
        palette={0: 'blue', 1: 'red'},
        alpha=0.6,
        legend='full'
    )

    # Randomly generate a new point
    new_point = np.random.rand(1, 2) * [max(X[:, 0]), max(X[:, 1])]  # Scale to fit the data range
    plt.scatter(new_point[0, 0], new_point[0, 1], c='green', s=100, label='New Point', edgecolors='black')

    # Add plot labels and legend
    plt.title("Scatter Plot with New Point")
    plt.xlabel(feature_columns[0])
    plt.ylabel(feature_columns[1])
    plt.legend(title="Diagnosis (0=Benign, 1=Malignant)")
    plt.show()

    # Train the KNN model using the training data
    while True:
        try:
            k = int(input("How many neighbors do you want to look at? (Positive integer)\n"))
            if k <= 1:
                print("The number of neighbors must be strictly positive!\n\n")
                continue
            elif k > max(5, sorted_db.shape[0]):
                print(f"There is only {max(5, sorted_db.shape[0])} neighbors ...\n\n")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer.\n\n")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Wait before showing the second plot
    print("\n\n\n\n\n\n\n\n\n\n")
    print("Classifying the new point...")
    sleep(5)

    # Evaluate the model accuracy using the testing set
    accuracy = knn.score(X_test, y_test)
    print("\n\n")
    print(f"Model Accuracy: {accuracy:.2f}")

    # Predict the label of the new point
    predicted_label = knn.predict(new_point)

    # Display the updated plot with the predicted color for the new point
    new_point_color = 'red' if predicted_label == 1 else 'blue'
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=sorted_db[feature_columns[0]],
        y=sorted_db[feature_columns[1]],
        hue=sorted_db['Target_Binary'],
        palette={0: 'blue', 1: 'red'},
        alpha=0.6,
        legend='full'
    )
    plt.scatter(new_point[0, 0], new_point[0, 1], c=new_point_color, s=100, edgecolors='black', label='New Point')
    plt.title("Scatter Plot with KNN Prediction")
    plt.xlabel(feature_columns[0])
    plt.ylabel(feature_columns[1])
    plt.legend(title="Diagnosis (0=Benign, 1=Malignant)")
    plt.show()

else:
    #K-Means
    # Plot the original data
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=sorted_db[feature_columns[0]],
        y=sorted_db[feature_columns[1]],
        color='black',
        alpha=0.6,
        legend='full'
    )

    # Add plot labels and title
    plt.title("All Data Points (Before Clustering)")
    plt.xlabel(feature_columns[0])
    plt.ylabel(feature_columns[1])

    # Show the plot
    plt.show()

    # Extract selected feature columns
    X = sorted_db[feature_columns].values

    # Apply K-Means clustering
    while True:
        try:
            n_clusters = int(input("Enter the number of clusters for K-Means: \n"))
            if n_clusters <= 1:
                print("The number of clusters must be strictly greater than 1!\n\n")
                continue
            elif n_clusters > max(5, sorted_db.shape[0]):
                print(f"There is only {max(5, sorted_db.shape[0])} points ...\n\n")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer greater than 1.\n\n")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    sorted_db['Cluster'] = kmeans.fit_predict(X)

    # Wait before showing the clustered plot
    print("\n\n\n\n\n\n\n\n\n\n")
    print("Clustering dataset...")
    sleep(1)

    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=feature_columns[0],
        y=feature_columns[1],
        hue='Cluster',
        data=sorted_db,
        palette='tab10',
        legend='full'
    )

    plt.title(f"K-Means Clustering: {feature_columns[0]} vs {feature_columns[1]}")
    plt.xlabel(feature_columns[0])
    plt.ylabel(feature_columns[1])
    plt.legend(title="Cluster")
    plt.show()

    # Display the cluster centroids
    centroids = kmeans.cluster_centers_
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=feature_columns[0],
        y=feature_columns[1],
        hue='Cluster',
        data=sorted_db,
        palette='tab10',
        legend='full'
    )
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, label='Centroids', marker='X')
    plt.title(f"K-Means Clustering with Centroids: {feature_columns[0]} vs {feature_columns[1]}")
    plt.xlabel(feature_columns[0])
    plt.ylabel(feature_columns[1])
    plt.legend(title="Cluster")
    plt.show()

    # Clustering End message
    print("\n\n\n\n\n\n\n\n\n\n")
    print("Clustering dataset done !")
    print("\n")

    # Calculate Silhouette Coefficient if there are more than 1 cluster
    if n_clusters > 1:
        silhouette_avg = silhouette_score(X, sorted_db['Cluster'])
        print(f"Silhouette Coefficient for {n_clusters} clusters: {silhouette_avg:.2f}")
    else:
        print("Silhouette Coefficient is not applicable for 1 cluster.")

    # Wait before the Elbow Method computation
    while True:
        go = input()
        if go != " ":
            continue
        break

    # Use the Elbow Method to determine the optimal number of clusters
    print("\n\nPerforming Elbow Method...")
    sse = []  # Sum of squared errors for each number of clusters
    max_k = min(10, sorted_db.shape[0])  # Maximum number of clusters to evaluate
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)  # Inertia: Sum of squared distances to centroids

    # Plot the Elbow Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), sse, marker='o', linestyle='-', color='blue')
    plt.title("Elbow Method for Optimal Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squared Errors (SSE)")
    plt.xticks(range(1, max_k + 1))
    plt.grid()
    plt.show()

    # Final message
    print("\n\n\n\n\n\n\n\n\n\n")
    print("Elbow curve ready !\n")
    print(f"In our case ({n_clusters} clusters) the Silhouette Coefficient is {silhouette_avg:.2f}")

