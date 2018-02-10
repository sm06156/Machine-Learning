from math import log
from collections import defaultdict
import math
from operator import itemgetter

from data import Point


class Tree:
    leaf = False # was True, Changed to false, probably better to keep as true, requires only 1 line instead of 2 w/e
    prediction = None
    feature = None
    threshold = None
    left = None
    right = None


def predict(tree, point):
    if tree.leaf:
        return tree.prediction
    i = tree.feature
    if (point.values[i] < tree.threshold):
        return predict(tree.left, point)
    else:
        return predict(tree.right, point)


def most_likely_class(prediction):
    labels = list(prediction.keys())
    probs = list(prediction.values())
    return labels[probs.index(max(probs))]


def accuracy(data, predictions):
    total = 0
    correct = 0
    for i in range(len(data)):
        point = data[i]
        pred = predictions[i]
        total += 1
        guess = most_likely_class(pred)
        if guess == point.label:
            correct += 1
    return float(correct) / total


def split_data(data, feature, threshold):
    left = []
    right = []
    # TODO: split data into left and right by given feature.
    # left should contain points whose values are less than threshold
    # right should contain points with values greater than or equal to
    for val in data:
        if threshold > val.values[feature]:
            left.append(val)
        else:
            right.append(val)

    return (left, right)


def count_labels(data):
    counts = {}
    # TODO: counts should count the labels in data
    # e.g. counts = {'spam': 10, 'ham': 4}
    for key in data:
        if isinstance(key,Point):
            counts[key.label] = counts.get(key.label, 0) + 1
        else: # dont even remember why I did this, fails if I remove, lol SOLID PROGRAMMING SKRILLZ
            counts[key[0]] = counts.get(key[0], 0) + 1


    return counts


def counts_to_entropy(counts):
    entropy = 0.0
    if len(counts) == 2:
        collegeCount = list(counts.values())[0]
        noCollegeCount = list(counts.values())[1]
        if collegeCount == 0 or noCollegeCount == 0 :
            return entropy
        total = sum(counts.values())
        entropy = -((collegeCount / total) * math.log2(collegeCount / total)) - (
                (noCollegeCount / total) * math.log2(noCollegeCount / total))
    return entropy


def get_entropy(data):
    counts = count_labels(data)
    entropy = counts_to_entropy(counts)
    return entropy

# This is an inefficient way to find the best threshold to maximize
# information gain.
def find_best_threshold(data, feature):
    entropy = get_entropy(data)
    best_gain = 0
    count_labels(data)
    total = len(data)
    best_threshold = None
    left = []
    right = []
    i  = 0
    calculatedValues = {}
    for val in data:
        if val in calculatedValues:
            continue
        calculatedValues[val.values[feature]] = val.values[feature]
        left, right = split_data(data, feature, val.values[feature])
        print(i)
        i += 1
        leftCount = len(left)
        rightCount = len(right)
        if leftCount == total or rightCount == total:
            continue
        temp = entropy - ((leftCount / total) * get_entropy(left)) - ((rightCount / total) * get_entropy(right))
        if temp > best_gain and temp != entropy:
            best_gain = temp
            best_threshold = val.values[feature]

    # TODO: Write a method to find the best threshold.
    return (best_gain, best_threshold)

def find_best_threshold_fast(data, feature):
    entropy = get_entropy(data)
    best_gain = 0
    # initialize dict and left/right lists
    rollingCount = {}
    left = []
    right = []
    #sort so we can do our poping from right/push to left
    for i in data:
        right.append([i.label, i.values[feature]])
    right = sorted(right,key=itemgetter(1))
    # get initial count of labels
    rollingCount["right"] = count_labels(right)
    listOfKeys = rollingCount["right"].keys()
    tempDict = {}
    count = 0
    #initialize left with keys and value of 0
    for i in listOfKeys:
        tempDict[i] = 0
        count += 1
    rollingCount["left"] = tempDict
    total = len(right)
    best_threshold = None

    for i in range(total):
        val = right[0]
        leftCount = len(left)
        rightCount = len(right)
        #Removed costly get_entropy call x2
        temp = entropy - ((leftCount / total) * counts_to_entropy(rollingCount["left"])) - ((rightCount / total) *
                                                                                        counts_to_entropy(rollingCount["right"]))
        if temp > best_gain and temp != entropy:
            best_gain = temp
            best_threshold = val[1]

        value = val[0]
        #append values from right list to left
        rollingCount["left"][value] += 1
        rollingCount["right"][value] -= 1
        left.append(val)
        right.pop(0)

    return (best_gain, best_threshold)


def find_best_split(data):
    if len(count_labels(data)) < 2:
        return None, None
    best_feature = None
    best_threshold = 0
    bestGain = 0
    # TODO: find the feature and threshold that maximize information gain.
    for i in range(0, len(data[0].values)):
        gain, temp_thres = find_best_threshold_fast(data, i)
        if bestGain <= gain:
            bestGain = gain
            best_threshold = temp_thres
            best_feature = i
    return (best_feature, best_threshold)



def make_leaf(data):
    tree = Tree()
    counts = count_labels(data)
    prediction = {}
    for label in counts:
        prediction[label] = float(counts[label]) / len(data)
    tree.prediction = prediction
    tree.leaf = True
    return tree


def c45(data, max_levels):

    tree = Tree()
    count = count_labels(data)
    prediction = {}

    if max_levels <= 0:
        return make_leaf(data)
    if (len(count) < 2 or list(count.values())[0] == list(count.values())[1]):
        return make_leaf(data)

    feat, thresh = find_best_split(data)
    left, right = split_data(data, feat, thresh)
    #Standard assignment of tree values
    tree.feature = feat
    tree.threshold = thresh
    tree.left = c45(left,max_levels-1)
    tree.right = c45(right, max_levels - 1)
    for label in count:
        prediction[label] = float(count[label]) / len(data)
    tree.prediction = prediction
    return tree


def submission(train, test):
    # TODO: Once your tests pass, make your submission as good as you can!
    tree = c45(train, 7)
    predictions = []
    for point in test:
        predictions.append(predict(tree, point))
    return predictions

# This might be useful for debugging.


def print_tree(tree):
    if tree.leaf:
        print("Leaf", tree.prediction)
    else:
        print("Branch", tree.feature, tree.threshold)
        print_tree(tree.left)
        print_tree(tree.right)
