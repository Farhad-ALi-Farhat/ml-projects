import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

def preprocess(df):
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def train(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2
    )

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, predictions))

    return model, vectorizer

def predict(model, vectorizer):
    msg = input("Enter message: ")
    vec = vectorizer.transform([msg])
    result = model.predict(vec)

    print("Spam" if result[0] == 1 else "Not Spam")

def main():
    df = load_data()
    df = preprocess(df)
    model, vectorizer = train(df)
    predict(model, vectorizer)

if __name__ == "__main__":
    main()
