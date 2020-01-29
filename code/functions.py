def class_labels(column):
    """
    Takes in target column and creates list of binary values. 1 (>=0.5) being in the 
    positive class (toxic), 0 (<0.5) being in the negative class (Not toxic)
    """
    
    class_label = []
    
    for row in column:
        
        if row < 0.5:
            class_label.append(0)
        else:
            class_label.append(1)
            
    return class_label

def clean_text(df, text):
    """
    Cleans text by replacing unwanted characters with blanks
    Replaces @ signs with word at
    Makes all text lowercase
    """
    
    df[text] = df[text].str.replace(r'[^A-Za-z0-9()!?@\s\'\`\*\"\_\n]', '')
    df[text] = df[text].str.replace(r'@', 'at')
    df[text] = df[text].str.lower()
    
    return df

def count_vectorizer(text_data):
    """
    Input: Text data
    Output: One hot encoded text data
    """
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    
    bag = cv.fit_transform(text_data)
    
    return bag, cv

def tf_idf(text_tokens):
    """
    Gives text data in tokenized form to create bag of words
    with TfidfVectorizer.  Returns bag of words and vectorizer.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    def dummy(doc):
        """
        Used since text data is already tokenized and preprocessed
        """
        return doc
    
    tf = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, token_pattern=None)
    
    bag = tf.fit_transform(text_tokens)
    
    return bag, tf

def plot_lsa(vectorized_words, class_labels):
    """
    Plots embedded words in two dimensions with color based on class
    """
    from sklearn.decomposition import TruncatedSVD
    
    # We are using 2 dimensions for visualization
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(vectorized_words)
    lsa_scores = lsa.transform(vectorized_words)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=4, c=class_labels, alpha=0.5)
   
    plt.show()

def lsa(vectorized_words, dimensions):
    """
    Takes in embedded text and number of components 
    wanted dimensionality reduced to
    """
    from sklearn.decomposition import TruncatedSVD
 
    lsa = TruncatedSVD(n_components=dimensions)
    lsa.fit(vectorized_words)
    lsa_values = lsa.transform(vectorized_words)
    
    return lsa_values, lsa

def evaluate_model(y_true, y_predicted):
    """
    performs cross validation of a model given training data by 
    taking the mean of 4 cross-validation scores
    """

    accuracy = accuracy_score(y_true, y_predicted)
    
    precision = precision_score(y_true, y_predicted)
    
    recall = recall_score(y_true, y_predicted)
    
    f1 = f1_score(y_true, y_predicted)
    
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('f1_score: ', f1)

def make_confusion_matrix(y_true, y_predicted):
    """
    Makes confusion matrix heatmap.
    y_true: True class values
    y_predicted: Predicted class values
    """

    confusion = confusion_matrix(y_true, y_predicted)
    plt.figure(dpi=80)
    sns.heatmap(confusion, cmap=plt.cm.winter, annot=True, square=True, fmt='d',
           xticklabels=['Irrelavant', 'Toxic'],
           yticklabels=['Irrelavant', 'Toxic']);
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

def roc_curve_plot(y_true, y_predicted_proba, y_predicted):
    """
    Creates roc curve given true y values and 
    a model's predicted probabilities and predicted class
    """

    fpr, tpr, thr = roc_curve(y_true, y_predicted_proba[:,1])
    auc = roc_auc_score(y_true, y_predicted)
    lw = 2
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label="Area Under Curve = %0.3f" % auc)
    plt.plot(y_true, y_true, color='green', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Plot')
    plt.legend(loc="lower right")
    plt.show()

def most_important_words(vectorizer, model, number):
    """
    Plots a horizontal bar graph of the top n most important words for a given vectorizer
    and a model fitted to data.
    number is number of words wanted
    """

    word_indices = {word:index for index,word in vectorizer.vocabulary_.items()}

# Stores each (word and importance) in a tuple that our model used for classifying each class
    word_importance = [[importance, word_indices[index]] for index,
                           importance in enumerate(model.coef_[0])]
    
# Sorts the list above so the words with the highest importance are first
    most_important_words = sorted(word_importance, key = lambda x: x[0], reverse=True)
    
    values = np.array(most_important_words)[:number, 0]
    words = np.array(most_important_words)[:number, 1]
    
    values_list = []
    for value in values:
        values_list.append(value)
    values_list.reverse()
    words_list = []
    for word in words:
        words_list.append(word)
    words_list.reverse()
    
    plt.figure(figsize=(8,6))
    plt.barh(words_list, values_list, align='center', alpha=0.5)
    plt.title('Important Words')
    plt.xticks(rotation=65)
    plt.show()