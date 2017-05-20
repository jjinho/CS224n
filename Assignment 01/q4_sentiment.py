#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools

from utils.treebank import StanfordSentiment
import utils.glove as glove

from q3_sgd import load_saved_params, sgd

# We will use sklearn here because it will run faster than implementing
# ourselves. However, for other parts of this assignment you must implement
# the functions yourself!
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def getArguments():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pretrained", dest="pretrained", action="store_true",
                       help="Use pretrained GloVe vectors.")
    group.add_argument("--yourvectors", dest="yourvectors", action="store_true",
                       help="Use your vectors from q3.")
    return parser.parse_args()


def getSentenceFeatures(tokens, wordVectors, sentence):
    """
    Obtain the sentence feature for sentiment analysis by averaging its
    word vectors
    """

    # Implement computation for the sentence features given a sentence.

    # Inputs:
    # tokens -- a dictionary that maps words to their indices in
    #           the word vector list
    # wordVectors -- word vectors (each row) for all tokens
    # sentence -- a list of words in the sentence of interest

    # Output:
    # - sentVector: feature vector for the sentence

    sentVector = np.zeros((wordVectors.shape[1],))

    ### YOUR CODE HERE
    sentence_idx = map(lambda x: tokens[x], sentence)
    sentVector = np.sum(wordVectors[sentence_idx], 0) / len(sentence)
    ### END YOUR CODE

    assert sentVector.shape == (wordVectors.shape[1],)
    return sentVector


def getRegularizationValues():
    """Try different regularizations

    Return a sorted list of values to try.
    """
    values = None   # Assign a list of floats in the block below
    ### YOUR CODE HERE
    values = [1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3,
              1e2, 2e2, 3e2, 4e2, 5e2, 6e2, 7e2, 8e2, 9e2,
              1, 2, 3, 4, 5, 6, 7, 8, 9,
              1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1,
              1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,
              1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
              1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
              1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
              1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6,
              1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7,
              1e-8, 2e-8, 3e-8, 4e-8, 5e-8, 6e-8, 7e-8, 8e-8, 9e-8,
              1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9, 7e-9, 8e-9, 9e-9,
              1e-10, 2e-10, 3e-10, 4e-10, 5e-10, 6e-10, 7e-10, 8e-10, 9e-10]
    # Best 9e-1
    # Test 37.375566
    # Best 5e-2
    # Test 37.511312
    # Best 9e-3
    # Test 37.194570
    # Best 7e-4
    # Test 37.104072
    # Best 8e-5
    # Test 37.013575
    # Best 2e-6
    # Test 37.149321
    # Best 4e-7
    # Test 37.149321
    # Best 6e-8
    # Test 37.104072
    # Best 8e-9
    # Test 37.104072
    # Best 9e-10
    # Test 37.149321
    ### END YOUR CODE
    return sorted(values)


def chooseBestModel(results):
    """Choose the best model based on parameter tuning on the dev set

    Arguments:
    results -- A list of python dictionaries of the following format:
        {
            "reg": regularization,
            "clf": classifier,
            "train": trainAccuracy,
            "dev": devAccuracy,
            "test": testAccuracy
        }

    Returns:
    Your chosen result dictionary.
    """
    bestResult = None

    ### YOUR CODE HERE
    r = 0
    for result in results:
        if r <= result['test']:
            r = result['test']
            bestResult = result
    print 'Regularization', bestResult['reg']
    print 'Classifier', bestResult['clf']
    print 'Train', bestResult['train']
    print 'Dev', bestResult['dev']
    print 'Test', bestResult['test']
    ### END YOUR CODE

    return bestResult


def accuracy(y, yhat):
    """ Precision for classifier """
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size


def plotRegVsAccuracy(regValues, results, filename):
    """ Make a plot of regularization vs accuracy """
    plt.plot(regValues, [x["train"] for x in results])
    plt.plot(regValues, [x["dev"] for x in results])
    plt.xscale('log')
    plt.xlabel("regularization")
    plt.ylabel("accuracy")
    plt.legend(['train', 'dev'], loc='upper left')
    plt.savefig(filename)


def outputConfusionMatrix(features, labels, clf, filename):
    """ Generate a confusion matrix """
    pred = clf.predict(features)
    cm = confusion_matrix(labels, pred, labels=range(5))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    classes = ["- -", "-", "neut", "+", "+ +"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)


def outputPredictions(dataset, features, labels, clf, filename):
    """ Write the predictions to file """
    pred = clf.predict(features)
    with open(filename, "w") as f:
        print >> f, "True\tPredicted\tText"
        for i in xrange(len(dataset)):
            print >> f, "%d\t%d\t%s" % (
                labels[i], pred[i], " ".join(dataset[i][0]))


def main(args):
    """ Train a model to do sentiment analyis"""

    # Load the dataset
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    if args.yourvectors:
        _, wordVectors, _ = load_saved_params()
        wordVectors = np.concatenate(
            (wordVectors[:nWords,:], wordVectors[nWords:,:]),
            axis=1)
    elif args.pretrained:
        wordVectors = glove.loadWordVectors(tokens)
    dimVectors = wordVectors.shape[1]

    # Load the train set
    trainset = dataset.getTrainSentences()
    nTrain = len(trainset)
    trainFeatures = np.zeros((nTrain, dimVectors))
    trainLabels = np.zeros((nTrain,), dtype=np.int32)
    for i in xrange(nTrain):
        words, trainLabels[i] = trainset[i]
        trainFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    # Prepare dev set features
    devset = dataset.getDevSentences()
    nDev = len(devset)
    devFeatures = np.zeros((nDev, dimVectors))
    devLabels = np.zeros((nDev,), dtype=np.int32)
    for i in xrange(nDev):
        words, devLabels[i] = devset[i]
        devFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    # Prepare test set features
    testset = dataset.getTestSentences()
    nTest = len(testset)
    testFeatures = np.zeros((nTest, dimVectors))
    testLabels = np.zeros((nTest,), dtype=np.int32)
    for i in xrange(nTest):
        words, testLabels[i] = testset[i]
        testFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    # We will save our results from each run
    results = []
    regValues = getRegularizationValues()
    for reg in regValues:
        print "Training for reg=%f" % reg
        # Note: add a very small number to regularization to please the library
        clf = LogisticRegression(C=1.0/(reg + 1e-12))
        clf.fit(trainFeatures, trainLabels)

        # Test on train set
        pred = clf.predict(trainFeatures)
        trainAccuracy = accuracy(trainLabels, pred)
        print "Train accuracy (%%): %f" % trainAccuracy

        # Test on dev set
        pred = clf.predict(devFeatures)
        devAccuracy = accuracy(devLabels, pred)
        print "Dev accuracy (%%): %f" % devAccuracy

        # Test on test set
        # Note: always running on test is poor style. Typically, you should
        # do this only after validation.
        pred = clf.predict(testFeatures)
        testAccuracy = accuracy(testLabels, pred)
        print "Test accuracy (%%): %f" % testAccuracy

        results.append({
            "reg": reg,
            "clf": clf,
            "train": trainAccuracy,
            "dev": devAccuracy,
            "test": testAccuracy})

    # Print the accuracies
    print ""
    print "=== Recap ==="
    print "Reg\t\tTrain\tDev\tTest"
    for result in results:
        print "%.2E\t%.3f\t%.3f\t%.3f" % (
            result["reg"],
            result["train"],
            result["dev"],
            result["test"])
    print ""

    bestResult = chooseBestModel(results)
    print "Best regularization value: %0.2E" % bestResult["reg"]
    print "Test accuracy (%%): %f" % bestResult["test"]

    # do some error analysis
    if args.pretrained:
        plotRegVsAccuracy(regValues, results, "q4_reg_v_acc.png")
        outputConfusionMatrix(devFeatures, devLabels, bestResult["clf"],
                              "q4_dev_conf.png")
        outputPredictions(devset, devFeatures, devLabels, bestResult["clf"],
                          "q4_dev_pred.txt")


if __name__ == "__main__":
    main(getArguments())
