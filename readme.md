# Dimensionality Reduction and Visualization App

This project contains Python scripts for performing dimensionality reduction using Principal Component Analysis (PCA) and visualizing the results through an interactive **Streamlit** web app.

---

## **File Descriptions**

- **`functions.py`**: Contains utility functions for data loading, aggregation, and filtering.
- **`PCA.py`**: Contains the implementation of a custom PCA function and an example using sklearn's PCA for comparison.
- **`StreamLit.py`**: Streamlit-based UI script for uploading a dataset, performing aggregation, and displaying PCA results interactively.

---

## **How to Run the Project**

### **1. Prerequisites**

Ensure you have Python 3.x installed and the following libraries installed:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

### **2. Running the Streamlit App**

1. Open a terminal/command prompt.
2. Navigate to the directory where the `StreamLit.py` file is located.
3. Run the Streamlit app using:

   ```bash
   streamlit run StreamLit.py
   ```

4. The app will open in your default web browser at `http://localhost:8501/`.

---

## **Using the Streamlit App**

### **1. Uploading a File**

- In the **sidebar**, click **"Choose a CSV or Excel file"** to upload your dataset (supports `.csv` and `.xlsx` formats).

### **2. Configuring Settings**

- **Column to Group By**: Select a column by which to group the data.
- **Aggregation Function**: Choose an aggregation function (`sum`, `mean`, `median`, `count`, etc.).
- **Number of PCA Components**: Enter the number of principal components (between 1 and 3 is recommended).
- **Data Display Mode**:
  - **City-wise**: Displays grouped data as-is.
  - **Party-wise**: Transposes the data.

### **3. Running the Analysis**

- Click **"Calculate & Display"** to run the PCA.
- The app will display:
  - **Data Preview Tab**: Shows the original uploaded dataset.
  - **PCA Results & Visualization Tab**:
    - **Grouped & Aggregated Data**: Shows the dataset after grouping and aggregation.
    - **PCA Output DataFrame**: Displays the PCA-reduced DataFrame.
    - **1D, 2D, or 3D PCA Visualization**: An interactive scatter plot visualizing the PCA.

### **4. Error Handling**

- If invalid inputs or file formats are provided, the app will show relevant error messages.

---

## **Running the PCA Script (Optional)**

You can also run `PCA.py` directly from the terminal to test the PCA implementation:

```bash
python PCA.py
```

This will display the PCA results for the example data in the console.

---

This README should help guide you through running the app and interacting with the UI. Let me know if you'd like any edits!

