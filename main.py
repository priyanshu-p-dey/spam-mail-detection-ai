import pandas as pd
import math
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv('data/spam.csv', encoding='latin-1', usecols=[0, 1])
df.columns = ['label', 'text']

# 2. Train-test split (80/20)
df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# 3. Tokenization & cleaning
for dataset in [df_train, df_test]:
   dataset['text'] = (
        dataset['text']
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.split()
    )

# 4. Count word frequencies
spam_counts = {}
ham_counts = {}


for row in df_train.itertuples(index=False):
    if row.label == 'spam':
        for w in row.text:
            spam_counts[w] = spam_counts.get(w, 0) + 1
    else:
        for w in row.text:
            ham_counts[w] = ham_counts.get(w, 0) + 1

# 5. FIXED: Calculation of Vocabulary Size (V)
all_train_words = np.concatenate(df_train['text'].values)
V = len(np.unique(all_train_words))

# 6. FIXED: Total word counts in each category (The denominators)
n_spam = sum(spam_counts.values())
n_ham = sum(ham_counts.values())

# 7. Priors based on message counts
label_counts = df_train['label'].value_counts()
total_messages = label_counts.sum()

p_spam = label_counts['spam'] / total_messages
p_ham = label_counts['ham'] / total_messages

# 8. Evaluation Loop
correct_pred = 0

for rows in df_test.itertuples(index=False):
    message = rows.text
    true_label = rows.label

    # Initialize with log of priors
    spam_score = math.log(p_spam)
    ham_score = math.log(p_ham)

    for w in message:
        # Use n_spam and n_ham + V for correct probability distribution
        p_w_spam = (spam_counts.get(w, 0) + 1) / (n_spam + V)
        p_w_ham = (ham_counts.get(w, 0) + 1) / (n_ham + V)

        spam_score += math.log(p_w_spam)
        ham_score += math.log(p_w_ham)

    prediction = 'spam' if spam_score > ham_score else 'ham'

    if prediction == true_label:
        correct_pred += 1
accuracy = (correct_pred / len(df_test)) * 100
print(f"Corrected Accuracy: {accuracy}%")
