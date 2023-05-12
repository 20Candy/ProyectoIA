import pickle

model1 = pickle.load(open('best_lr1.sav', 'rb'))
vectorizer1 = pickle.load(open('vectorizer1.sav', 'rb'))

model2 = pickle.load(open('best_lr2.sav', 'rb'))
vectorizer2 = pickle.load(open('vectorizer2.sav', 'rb'))

while True:
    newTweetText = input("Ingrese un tweet ('exit' para salir): ")
    if newTweetText == 'exit':
        break

    newTweet = vectorizer1.transform([newTweetText]).toarray()
    result = model1.predict(newTweet)

    if result[0] == "cyberbullying":
        print("El tweet es cyberbullying")

        newTweet = vectorizer2.transform([newTweetText]).toarray()
        result = model2.predict(newTweet)

        print("Tipo de cyberbullying: ", result)
    else:
        print("El tweet no es cyberbullying")