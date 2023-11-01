import math
import os
from pathlib import WindowsPath
import subprocess

dir_test_pos = r"C:\Users\Connor_Laptop\Documents\GitHub\CSC381\HomeWork2\movie-review-HW2\aclImdb\test\pos"
dir_test_neg = r"C:\Users\Connor_Laptop\Documents\GitHub\CSC381\HomeWork2\movie-review-HW2\aclImdb\test\neg"
dir_train_pos = r"C:\Users\Connor_Laptop\Documents\GitHub\CSC381\HomeWork2\movie-review-HW2\aclImdb\train\pos"
dir_train_neg = r"C:\Users\Connor_Laptop\Documents\GitHub\CSC381\HomeWork2\movie-review-HW2\aclImdb\train\neg"



# Function to preprocess files
def preprocess_files(path):
    script_path = "C:\\Users\\Connor_Laptop\\Documents\\GitHub\\Naive-Bayes\\pre-process.py"
    subprocess.call(["python3", script_path, path])

# Function to get a list of files in a directory and preprocess them
def get_files(path, filesarr):
    for x in os.listdir(path):
        if os.path.isfile(os.path.join(path, x)):
            if(preprocess_flag):
                preprocess_files(os.path.join(path, x))
            filesarr.append(os.path.join(path, x))
    return filesarr

# Function to count words in documents
def count_words(word, count):
    if word in count:
        count[word] += 1
    else:
        count[word] = 1
    return count

# Function to get Bag of Words features from files
def get_BOW_features(filesarr):
    words = {}
    for x in filesarr:
        file = open(x, errors="ignore")
        lines = file.readlines()
        for sentence in lines:
            for word in sentence.split():  # Split sentence into words
                words = count_words(word, words)
        file.close()
    return words

# Function to train the Naive Bayes classifier
def train_NB(posfiles, negfiles):
    global poswords
    global negwords
    global posprob
    global negprob
    poswords = get_BOW_features(posfiles)
    negwords = get_BOW_features(negfiles)
    posprob = len(posfiles) / (2 * len(posfiles) + len(negfiles))
    negprob = len(negfiles) / (2* len(posfiles) + len(negfiles))
    
    # You can calculate probabilities and other training steps here

# Function to test the Naive Bayes classifier
def test_NB():
    correct_predictions = 0
    total_predictions = 0
    global poswords
    global negwords
    global posprob
    global negprob

    for test_dir, label in [(dir_test_pos, 'pos'), (dir_test_neg, 'neg')]:
        test_files = get_files(test_dir, [])
        for test_file in test_files:
            total_predictions += 1
            file = open(test_file, errors="ignore")
            lines = file.readlines()
            word_scores = {'pos': 0, 'neg': 0}

            for sentence in lines:
                for word in sentence.split():
                    if word in poswords:
                        word_scores['pos'] += math.log((poswords[word] + 1) / (len(poswords) + len(negwords)))
                    if word in negwords:
                        word_scores['neg'] += math.log((negwords[word] + 1) / (len(negwords) + len(poswords)))

            # Add log probabilities of class priors (posprob and negprob)
            word_scores['pos'] += math.log(posprob)
            word_scores['neg'] += math.log(negprob)

            # Assign the class with the highest log probability
            predicted_label = max(word_scores, key=word_scores.get)

            if predicted_label == label:
                correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Call the test_NB function in your main function
def main():
    global posfiles
    global negfiles
    global preprocess_flag
    global train_flag
    global test_flag
    print("Would you like to preprocess your files? (y/n)")
    if(input() == "y"):
        preprocess_flag = True
    if(input() == "n"):
        preprocess_flag = False
    else:
        print("Please enter y or n")
    print("Would you like to retrain your model? (y/n)")
    if(input() == "y"):
        train_flag = True
    if(input() == "n"):
        train_flag = False
    else:
        print("Please enter y or n")
    print("Would you like to test your model on a custom input? (y/n)")
    if(input() == "y"):
        test_flag = True
    if(input() == "n"):
        test_flag = False
    else:
        print("Please enter y or n")
    posfiles = get_files(dir_train_pos, [])
    negfiles = get_files(dir_train_neg, [])
    train_NB(posfiles, negfiles)
    test_NB()
    
if __name__ == "__main__":
    main()

