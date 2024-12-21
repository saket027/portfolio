# Summary
This is the summary of the project.


# Detailed Summaries for Each Step and Experiment

Below are **detailed summaries** for **Steps 1–9** and **Experiments 1–7** in the **Air Quality Prediction Project**. Each summary is  providing insight into the goals, methods, and outcomes of each phase.

---------
#### **Links:**
- **Streamlit App:** <a href="https://airqualityapp-czxh5jumzvuy2cn5obtcag.streamlit.app/" target="_blank">Air Quality Prediction</a>
- **DagsHub:** <a href="https://dagshub.com/saket027/EAS_503_Updated_Air_Quality_Prediction/experiments" target="_blank">Experiments</a>
- **DockerHub:** <a href="https://hub.docker.com/repository/docker/saket027/airquality_app/general" target="_blank">Docker Image</a>
- **Final Deployed Model:** GradientBoostingClassifier
--------
## **Step 1: Create a Normalized Database**

Step 1 establishes a structured foundation for the project by **normalizing** the raw pollution dataset into a 3NF relational schema. Rather than keeping everything in a single CSV, you split the data into logical entities—such as `Location` (for attributes like `Proximity_to_Industrial_Areas`, `Population_Density`) and `AirQuality` (for metrics like `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, and the target variable `Air Quality`). A link table, often called `LocationAirQuality`, matches each `Location_ID` with its corresponding `Measurement_ID`, ensuring that the relationship between geographic/demographic data and pollution metrics is clearly defined and free of redundancy.

You implement this using Python’s built-in `csv` library (so as not to rely on Pandas in this step) and store the tables in a **SQLite** database, generating a lightweight `.db` file. The approach drastically reduces data repetition; each unique location appears only once, and each unique measurement set likewise appears only once. This guards against inconsistent updates that might occur if the same location details were repeated in multiple rows. It also streamlines querying in future steps, letting you easily reassemble data with SQL joins.

Ultimately, **Step 1** transforms a single flat CSV into multiple tables that respect database normalization rules. This process paves the way for a more maintainable, extensible data setup. If you add new measurement types or new location attributes, the normalized structure accommodates these changes without bloating the dataset. By enforcing relationships strictly, you lay a solid groundwork for all subsequent data exploration and machine learning activities in the project.

---

## **Step 2: SQL Join and Loading Data into Pandas**

Having stored the data in normalized tables, **Step 2** demonstrates how to **reconstruct** a single comprehensive dataset via SQL `JOIN` queries. By querying the SQLite database—where your `Location` and `AirQuality` tables are related through `LocationAirQuality`—you effectively piece together each location record with its corresponding pollution metrics in a single query result. This approach highlights the benefits of normalization: you maintain clean, integrity-safe tables internally, but can still merge them into the “flat” shape that data scientists prefer for exploration and modeling.

After executing the multi-table JOIN, you read the result directly into a Pandas DataFrame using `pd.read_sql_query()`. The DataFrame features key columns like `Temperature`, `PM2_5`, `Population_Density`, and so on, now merged into one table. This structure is more convenient for tasks like EDA and machine learning. You further persist this DataFrame to a new CSV file—commonly named `data_from_step_2.csv`—ensuring that subsequent steps can simply read the CSV instead of rerunning SQL queries each time.

In short, Step 2 seamlessly transitions from a relational model back to a data-scientist-friendly tabular format. It proves that normalization doesn’t hamper usability. Instead, it provides a strong data integrity layer while you remain free to produce consolidated DataFrames on demand. The final CSV is a critical checkpoint: if further transformations are required later, you can start from this single, consistent file, preserving the benefits of the normalized schema and ensuring reusability in every phase of your project.

---

## **Step 3: Exploratory Analysis & Train/Test Split**

Step 3 introduces **exploratory data analysis** and splitting your dataset into training and testing subsets. By loading `data_from_step_2.csv` into Pandas, you can readily check basic statistics and class distributions for the target variable, `Air_Quality`. Visualizing the counts of each class, such as Good, Moderate, Poor, and Hazardous, clarifies whether your dataset is imbalanced—an essential insight because it may dictate whether a stratified split is necessary.

If the dataset is moderately or heavily imbalanced, you apply a **stratified** train/test split, ensuring each class ratio in training roughly matches that of the overall dataset. Such balance is crucial in preventing your model from ignoring minor classes and inflating performance metrics artificially. You verify the distribution by examining numeric percentages or by plotting bar charts for both the train and test subsets, ensuring they closely resemble the original distribution. This step thus guards against poor generalization and inconsistent model evaluation.

At the same time, you may spot interesting relationships or anomalies in the dataset. Basic descriptive statistics or histograms can reveal unusual outliers (for example, extremely high PM2.5 values) or missing data patterns. Although deeper EDA (like correlation checks) arrives in later steps, Step 3 forms your initial understanding of how balanced or skewed the data is, letting you isolate a stable training portion for model development and a pristine testing subset for unbiased performance measurement. This organized structure sets the stage for fair, reliable evaluation in your subsequent modeling efforts.

---

## **Step 4: Data Exploration with yProfile & Correlation Analysis**

Step 4 elevates your exploration by utilizing **ydata-profiling** (formerly Pandas Profiling) to produce a comprehensive EDA report. Rather than individually calculating missing values or distribution histograms, you rely on the profiling tool’s automatic generation of summary statistics, correlations, and outlier detection. This single HTML (or Jupyter-based) report reveals everything from each column’s mean, median, and standard deviation, to a deep look at potential outliers and weird data patterns.

Simultaneously, you create a **correlation matrix**—often visualized in a Seaborn heatmap—to highlight pairs of features that are strongly or weakly correlated. High correlation could mean redundant features (e.g., PM2.5 and PM10) that might be combined or one dropped entirely. Low correlation with the target can prompt you to question whether a feature significantly contributes to predictive power. These insights from correlation checks help shape your approach to feature selection or dimensionality reduction in later experiments.

Additionally, the profiling tool can unearth data anomalies you might otherwise miss: non-numeric entries where numeric data is expected, columns with a surprisingly high missing rate, or distributions that are heavily skewed. Armed with this knowledge, you can correct or transform such data before feeding it into machine learning algorithms. Ultimately, Step 4 ensures you have a detailed blueprint of how your dataset behaves, guiding strategic decisions around transformations, selection, and engineering in future modeling steps.

---

## **Experiment 1: Baseline Logistic Regression**

Experiment 1 sets the foundation by testing a **Logistic Regression** model as a baseline, bundled within a pipeline of basic transformations. Typically, you apply scaling (like StandardScaler), handle categorical features (OneHotEncoder), and possibly do a simple log transform to mitigate right-skewed distributions. The key objective is to see how a straightforward linear classifier fares on your dataset in terms of F1 scores (macro and micro).

You track performance by running cross-validation—often 10-fold—logging metrics to MLflow. The rationale behind choosing logistic regression first is twofold: (1) it’s computationally fast, offering quick feedback; (2) it’s linear, providing a minimal baseline to surpass. If you discover that logistic regression already achieves reasonably high F1 scores, you know your dataset is linearly separable to some extent. Conversely, if results are weak, you gain insight that more advanced methods (tree-based or boosting) may be needed.

Another crucial aspect is analyzing confusion matrices or class-based metrics. Logistic regression might excel at common classes but struggle with rarer ones, highlighting whether your data is truly balanced. By logging these results systematically in MLflow, you can compare them to subsequent experiments with more intricate transformations or classifiers. This baseline approach ensures a consistent reference point, reminding you how simple solutions measure up against more complex pipelines. If new experiments only bring marginal gains, it might indicate that your baseline was already quite competent—or that fundamental data issues remain.

---

## **Experiment 2: Multiple Classifiers**

In Experiment 2, you expand your scope by testing a range of classifiers under a shared preprocessing pipeline. Typically, you keep the transformations from Experiment 1—like scaling, log transforms, or one-hot encoding—consistent, then swap out the final classifier step for RidgeClassifier, RandomForestClassifier, XGBClassifier, or any others you wish to explore. This approach highlights how different algorithmic families tackle your data differently, from linear models (RidgeClassifier) to ensemble trees (RandomForest, XGBoost).

By logging each classifier’s F1-macro and F1-micro scores in MLflow, you gain an immediate comparison. Perhaps XGBoost outperforms the linear methods, or RandomForest exhibits strong performance but is slower to train. These insights shape your hypothesis about whether your dataset’s relationships are linear, heavily non-linear, or require robust handling of outliers.

Another advantage is seeing how each classifier responds to the same transformations. If a classifier performs poorly, you can check if certain features or transformations are incompatible. Or you might discover that some classifiers (like tree-based ones) are less sensitive to scaling or logs, so the transformations might matter less. The experiment also guides your later steps regarding hyperparameter tuning. For instance, if random forests are best but not fully optimized, you can deepen your search in subsequent experiments. By casting a broad net in Experiment 2, you capture a wide array of potential solutions, letting you quickly zero in on the most promising approach for more focused experimentation.

---

## **Experiment 3: Feature Engineering**

Experiment 3 tackles **feature engineering**—the process of creating or combining attributes to better represent patterns in the data. You might form ratios like `PM_Ratio = PM2.5 / (PM10 + 1)`, or generate interaction terms like `Temperature * Humidity`. Each new feature attempts to highlight relationships that might be hidden when data is left in raw form. If PM2.5 and PM10 track together, building a ratio or difference might isolate a more meaningful signal.

In your pipeline, you insert a custom transformer that, before any scaling or encoding, calculates these new columns and appends them to the dataset. Following the transformations, you apply the same classifier steps as before—perhaps logistic regression or random forest—to see if these engineered features boost F1 scores. You log the results in MLflow, comparing them to the baseline experiments that lacked these specialized features.

Sometimes, you’ll see a significant lift in performance from a single well-crafted ratio or difference. Other times, multiple minor features collectively raise the score by a small margin. Or you might see no improvement if the new features add noise or replicate existing signals. Regardless, feature engineering is a potent lever, especially in tabular data tasks. By systematically testing the impact in Experiment 3, you gain clarity on which features genuinely aid classification. If results are promising, you might refine them further in subsequent experiments or combine them with different transformations, seeking that perfect synergy for higher predictive power.

---

## **Experiment 4: Feature Selection**

Experiment 4 concentrates on **feature selection**, aiming to reduce the dimensionality of your dataset and filter out uninformative or redundant attributes. You employ multiple strategies, such as a variance threshold to drop nearly constant columns, correlation-based filtering to remove highly correlated pairs, and a model-based selection (e.g., a RandomForest’s feature importances). The objective is twofold: (1) reduce model overfitting by eliminating noisy or irrelevant features, and (2) potentially speed up training and inference.

You integrate these steps into a pipeline. For example, you might apply `VarianceThreshold` first, then a custom correlation filter, and finally a `SelectFromModel` stage that uses a random forest to rank feature importance. After this selection, the pruned set of features moves on to a classifier (like logistic regression). By logging F1 scores, you see if you can improve or at least maintain performance with fewer features. Sometimes, removing extraneous variables clarifies the signal for the model, leading to higher accuracy.

Experiment 4 clarifies how your data’s complexity relates to your model’s generalization capabilities. If performance increases or stays the same with fewer features, that’s a win, as it means your solution is simpler and less prone to overfit. If performance drops, it may suggest that certain features presumed unimportant were actually valuable. The synergy between correlation checks, variance thresholds, and model-based importance forms a robust combination for trimming down your feature set. Ultimately, you can confirm that streamlining attributes can be just as pivotal as adding new ones.

---

## **Experiment 5: PCA**

In Experiment 5, you investigate **Principal Component Analysis (PCA)** to see whether dimensionality reduction helps your model. PCA identifies orthogonal axes (principal components) that capture the greatest variance in the data. You typically run it after basic transformations—like scaling or log transforms—so the principal components are derived from consistent numeric distributions. A scree plot visualizes how each component’s explained variance tapers off, letting you choose a cutoff that captures, say, 95% of the variance. 

By training a classifier (maybe logistic regression) on just these selected principal components, you measure if performance improves or remains stable. Sometimes, reducing dimensions can remove noise or correlated features, leading to a simpler model that generalizes better. In other cases, you might see performance dip slightly, since PCA is purely unsupervised and may discard class-related variance. However, even a small drop in F1 might be worth it if the model speeds up significantly or if fewer features means fewer resources in production.

You log each run in MLflow, comparing F1 scores before and after PCA. Additionally, you observe how many components were chosen (`n_components`). This approach fosters a numeric approach to dimensionality decisions—rather than guesswork. If you find that 10 components suffice to match prior performance, you might adopt that as your new baseline. If the best explained variance threshold leads to only modest changes, you can weigh the trade-offs in training speed or interpretability. Overall, Experiment 5 clarifies whether PCA-driven dimensionality reduction is beneficial for your specific air quality dataset.

---

## **Experiment 6: Polynomial Features + Gradient Boosting**

Experiment 6 tests a more advanced, **custom approach**. By adding polynomial features in your pipeline, you capture non-linear interactions among numeric variables—like `(PM2.5)^2`, `Temperature * Humidity`, or `(CO)^2`. Such expansions can uncover complex patterns that linear transformations miss. Then, you pair these polynomial features with a **GradientBoostingClassifier**, a powerful ensemble method known for boosting performance on structured data. 

The hypothesis is that the synergy of polynomial expansions plus gradient boosting could yield higher F1 scores than earlier experiments. After integrating these steps—scaling, polynomial expansions, and gradient boosting—into a pipeline, you log cross-validation results in MLflow. The new features might significantly improve model capacity to separate classes, especially if your original dataset had non-linear relationships that simpler transformations didn’t exploit.

You watch the F1-macro and F1-micro means to see if polynomial expansions add genuine value. Sometimes, they do wonders for non-linear classification tasks; other times, they add too many features and risk overfitting. But because you store the entire pipeline as an artifact, if the experiment shows a strong boost in performance, it becomes a prime candidate for final selection. Additionally, you may tune parameters in the GradientBoostingClassifier—like number of estimators or learning rate—to refine results. Experiment 6’s success typically signals that polynomial interactions, combined with an advanced ensemble method, can lead to robust predictions in your air quality scenario.

---

## **Experiment 7: Mutual Information + LightGBM**

Experiment 7 tries **another custom approach** using mutual information-based feature selection plus **LightGBM**. Mutual information evaluates how much knowing a given feature reduces uncertainty about the target class, thus highlighting the top K most relevant features. After this selection step, you pass the reduced set of features to a `LGBMClassifier`—a gradient boosting library often praised for fast performance and strong results.

Because LightGBM can handle large data efficiently and incorporate built-in handling for various feature types, it pairs nicely with a strategic feature selection approach. By dropping features that provide minimal mutual information, you streamline the model’s training to only the most informative variables—like certain pollution metrics strongly tied to the target. You then measure if the synergy of mutual information and LightGBM yields higher F1 scores relative to previous experiments.

In typical MLflow logging fashion, you compare F1-macro and F1-micro. If results jump significantly, it suggests your dataset had extraneous columns that overshadowed key signals, and that LightGBM’s fast boosting approach capitalized on the refined feature set. On the other hand, if performance remains similar, it might indicate that LightGBM could handle those extraneous columns anyway, or that your top K features alone weren’t enough. Nonetheless, this experiment demonstrates a more targeted feature selection method, pivoting from purely correlation-based or random forest feature importance to a direct measure of mutual information with the target. If LightGBM with selected features stands out, it can become a prime contender for final model selection.

---

## **Step 8: Compare Experiments & Pick the Best Model**

Step 8 unifies all prior experiments by querying MLflow for each experiment’s best run. Typically, you define “best” by the highest `f1_macro_mean` or another chosen metric. You compile these top runs into a table or bar chart, letting you visually compare, for instance, how polynomial expansions with gradient boosting (Experiment 6) stacks up against mutual information + LightGBM (Experiment 7). Once you identify the single best run overall—perhaps a run from Experiment 6 with a certain pipeline configuration—you load that run’s model from MLflow, ensuring no guesswork.  

You finalize your choice by **saving** the loaded pipeline as `final_model.joblib` locally, bridging ephemeral experimentation and permanent deployment. This step underscores the advantage of tracking everything in MLflow: you have a reproducible record of which transformations and hyperparameters led to the best result. If there’s a tie, you might weigh training speed, interpretability, or confusion matrix specifics for certain classes.  

Ultimately, Step 8 cements the entire modeling phase, producing a single artifact that stands above the rest in performance. By enumerating each experiment’s champion run, you confirm the methodical nature of your experimentation—no single approach was overlooked or lost. This fosters confidence that the selected model indeed harnesses the best combination of transformations, feature engineering, and classifier choice. Step 8 thus readies you for final deployment steps, ensuring you rely on the truly best pipeline discovered among all your trials.

---

## **Step 9: Deployment & Serving the Final Model**

Step 9 completes the journey by **deploying** your chosen best model as a production-ready service. You begin by writing a FastAPI application (`main.py`) that loads your locally saved `final_model.joblib`. The app defines a `/predict` endpoint, which expects JSON data matching your feature schema. When a request arrives, the code transforms the input into a DataFrame, calls `model.predict()`, and returns the predicted class label (e.g., Good, Moderate, Poor, Hazardous). This effectively turns your pipeline—complete with scaling, encoding, or advanced transformations—into a REST API anyone can query.

```{tableofcontents}
```