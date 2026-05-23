import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

fake_df = pd.read_csv(r"C:\Users\Hanumakshi\Downloads\archive (5)\Fake.csv", engine='python', on_bad_lines='skip')
fake_df["target"] = "fake"

true_df = pd.read_csv(r"C:\Users\Hanumakshi\Downloads\archive (5)\True.csv", engine='python', on_bad_lines='skip')
true_df["target"] = "real"

# Add labels
fake_df["target"] = 0
true_df["target"] = 1

# Combine datasets
df = pd.concat([fake_df, true_df])

# Text preprocessing function
def clean_text(text):

    text = text.lower()  # lowercase

    text = re.sub(r'http\S+', '', text)  # remove links

    text = re.sub(r'\d+', '', text)  # remove numbers

    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation

    return text

# Apply preprocessing
df["clean_text"] = df["text"].apply(clean_text)

# Show output
#print(df[["text", "clean_text"]].head())

X = df["text"].str.lower()
y = df["target"]

tfidf = TfidfVectorizer(stop_words='english', max_features=9000)
X_vect = tfidf.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vect, y)

news_input =  "Aliens have officially signed a peace agreement with world leaders in Switzerland."

vect_input = tfidf.transform([ "Aliens have officially signed a peace agreement with world leaders in Switzerland.".lower()])
prediction = model.predict(vect_input)[0]
probability = model.predict_proba(vect_input)[0][prediction]

if prediction == 1:
    result = "Real News"
else:
    result = "Fake News"

print(f"The news is: {result} with a probability of {probability:.2f}")

import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data for plotting probabilities
prob_fake = probability if prediction == 0 else (1 - probability)
prob_real = probability if prediction == 1 else (1 - probability)

prob_df = pd.DataFrame({
    'News Type': ['Fake News', 'Real News'],
    'Probability': [prob_fake, prob_real]
})

plt.figure(figsize=(7, 5))
sns.barplot(x='News Type', y='Probability', data=prob_df, hue='News Type', palette=['salmon', 'lightgreen'], legend=False)
plt.title('Prediction Probability for Input News')
plt.xlabel('News Type')
plt.ylabel('Probability')
plt.ylim(0, 1) # Ensure y-axis goes from 0 to 1 for probabilities
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualization
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

