📈 Logistic Regression – Bank Marketing Dataset
This project applies Logistic Regression on a real-world bank marketing dataset.
It demonstrates the full preprocessing pipeline and a clean implementation of training preparation steps.

✍️ Author Note
I created this project a few months ago while learning foundational ML, including Logistic Regression, SMOTE balancing, and one-hot encoding—all manually coded instead of relying on high-level wrappers.

Inspired by lessons from Andrej Karpathy and Andrew Ng.

🧠 Project Highlights
📊 Dataset: Bank Marketing Data (CSV)
🏷️ Target: Whether a client subscribed to a term deposit (y)
🔢 Preprocessing:
Manual one-hot encoding (with NumPy + mapping)
SMOTE balancing for imbalanced binary classes
Standard normalization
🧪 Data split: 4000 samples for training, rest for testing
📁 File Structure
.
├── LogisticRegression_Cleaned_For_GitHub.ipynb
├── data/
│   └── bank_refined.csv  # <- Place your dataset here
📦 Requirements
pip install numpy pandas matplotlib imbalanced-learn
💡 Sample Code Snippet
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
🚀 Future Enhancements
Add model training + evaluation cells
Accuracy/Confusion Matrix plotting
ROC Curve visualization
📚 Reference
Andrej Karpathy's Micrograd
Andrew Ng’s ML Specialization
