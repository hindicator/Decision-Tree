import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Fill in the implementation details of the class DecisionTree using this file.
 * Any methods or
 * secondary classes that you want are fine but we will only interact with those
 * methods in the
 * DecisionTree framework.
 * 
 * You must add code for the 1 member and 4 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
  private DecTreeNode root;
  // ordered list of class labels
  private List<String> labels;
  // ordered list of attributes
  private List<String> attributes;
  // map to ordered discrete values taken by attributes
  private Map<String, List<String>> attributeValues;

  /**
   * Answers static questions about decision trees.
   */
  DecisionTreeImpl() {
    // no code necessary this is void purposefully
  }

  /**
   * Build a decision tree given only a training set.
   * 
   * @param train: the training set
   */
  DecisionTreeImpl(DataSet train) {
    this.labels = train.labels;
    this.attributes = train.attributes;
    this.attributeValues = train.attributeValues;
    this.root = _buildDecisionTree(train.instances, train.attributes, train.instances, null);
  }

  @Override
  public String classify(Instance instance) {
    DecTreeNode currentNode = root;
    while (!currentNode.terminal) {
      for (DecTreeNode child : currentNode.children) {
        if (instance.attributes.get(getAttributeIndex(currentNode.attribute))
            .equals(child.parentAttributeValue)) {
          currentNode = child;
          break;
        }
      }
    }
    return currentNode.label;
  }

  @Override
  public void rootInfoGain(DataSet train) {
    this.labels = train.labels;
    this.attributes = train.attributes;
    this.attributeValues = train.attributeValues;
    for (String attribute : attributes) {
      System.out.format("%s %.5f\n", attribute, infoGain(attribute, train.instances));
    }
  }

  @Override
  public void printAccuracy(DataSet test) {
    System.out.format("%.5f\n", this.getAccuracy(test.instances));
  }

  /**
   * Get the accuracy of the decision tree on a given test set.
   * 
   * @param test: the test set
   * @return the accuracy
   */
  private double getAccuracy(List<Instance> examples) {
    int correct = 0;
    for (Instance instance : examples) {
      if (classify(instance).equals(instance.label)) {
        correct++;
      }
    }
    return (double) correct / examples.size();
  }

  /**
   * Build a decision tree given a training set then prune it using a tuning set.
   * ONLY for extra credits
   * 
   * @param train: the training set
   * @param tune:  the tuning set
   */
  DecisionTreeImpl(DataSet train, DataSet tune) {
    this.labels = train.labels;
    this.attributes = train.attributes;
    this.attributeValues = train.attributeValues;

    this.root = _buildDecisionTree(train.instances, train.attributes, train.instances, null);
    pruneTree(this.root, tune);
  }

  /**
   * Prune the decision tree using the given tuning set.
   * 
   * @param node the current node being pruned
   * @param tune
   */
  private void pruneTree(DecTreeNode node, DataSet tune) {
    if (node.terminal) {
      return;
    }

    for (DecTreeNode child : node.children) {
      this.pruneTree(child, tune);
      double accuracyWithoutPruning = getAccuracy(tune.instances);
      node.terminal = true;

      double accuracyWithPruning = getAccuracy(tune.instances);
      node.terminal = false;

      // If accuracy is better or the same with pruning, keep it.
      if (accuracyWithPruning >= accuracyWithoutPruning) {
        node.terminal = true;
      }
    }
  }

  @Override
  /**
   * Print the decision tree in the specified format
   */
  public void print() {
    printTreeNode(root, null, 0);
  }

  /**
   * Prints the subtree of the node with each line prefixed by 4 * k spaces.
   */
  public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < k; i++) {
      sb.append("    ");
    }
    String value;
    if (parent == null) {
      value = "ROOT";
    } else {
      int attributeValueIndex = this.getAttributeValueIndex(parent.attribute, p.parentAttributeValue);
      value = attributeValues.get(parent.attribute).get(attributeValueIndex);
    }
    sb.append(value);
    if (p.terminal) {
      sb.append(" (" + p.label + ")");
      System.out.println(sb.toString());
    } else {
      sb.append(" {" + p.attribute + "?}");
      System.out.println(sb.toString());
      for (DecTreeNode child : p.children) {
        printTreeNode(child, p, k + 1);
      }
    }
  }

  /**
   * Helper function to get the index of the attribute in attributes list
   */
  private int getAttributeIndex(String attr) {
    for (int i = 0; i < this.attributes.size(); i++) {
      if (attr.equals(this.attributes.get(i))) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Helper function to get the index of the attributeValue in the list for the
   * attribute key in the attributeValues map
   */
  private int getAttributeValueIndex(String attr, String value) {
    for (int i = 0; i < attributeValues.get(attr).size(); i++) {
      if (value.equals(attributeValues.get(attr).get(i))) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Helper function to get the attribute value for the attribute at the given
   * 
   * @param attribute
   * @param examples
   * @return double
   */
  private double infoGain(String attribute, List<Instance> examples) {
    double attributeEntropy = 0;
    for (String value : attributeValues.get(attribute)) {
      List<Number> temp = countRowsWithAttributeValue(examples, attribute, value);
      attributeEntropy += entropy(temp, examples.size());
    }

    return calculateClassEntropy(examples) - attributeEntropy;
  }

  /**
   * Helper function to calculate entropy
   */
  private double entropy(List<Number> values, Number totalRows) {
    double totalEntropy = 0;
    Number totalRowsWithAttributeValue = values.stream().reduce(0, (a, b) -> a.intValue() + b.intValue());
    double attributeRatio = totalRowsWithAttributeValue.doubleValue() / totalRows.doubleValue();

    if (totalRowsWithAttributeValue.intValue() == 0) {
      return 0;
    }

    for (Number value : values) {
      if (value.intValue() == 0) {
        continue;
      }
      double attributeValueLabelRatio = value.doubleValue() / totalRowsWithAttributeValue.doubleValue();
      totalEntropy += attributeValueLabelRatio * Math.log(attributeValueLabelRatio) / Math.log(2);
    }

    return attributeRatio * totalEntropy * -1;
  }

  /**
   * Helper function to count the number of rows with the given attribute value
   * Count how many rows have the given attributeValue for the given attribute for
   * each attributeValue
   * 
   * @param examples
   * @param attribute
   * @param value
   * @return List<Number>
   */
  private List<Number> countRowsWithAttributeValue(List<Instance> examples, String attribute, String value) {
    List<Number> count = new ArrayList<Number>();
    this.labels.forEach(label -> count.add(0));
    int attributeIndex = getAttributeIndex(attribute);
    for (Instance instance : examples) {
      if (instance.attributes.get(attributeIndex).equals(value)) {
        int labelIndex = this.labels.indexOf(instance.label);
        count.set(labelIndex, count.get(labelIndex).intValue() + 1);
      }
    }
    return count;
  }

  /**
   * Helper function to calculate class entropy
   * 
   * @param examples
   * @return double
   */
  private double calculateClassEntropy(List<Instance> examples) {
    List<Number> count = new ArrayList<Number>();
    double totalRows = examples.size();
    double classEntropy = 0;
    this.labels.forEach(label -> count.add(0));

    for (Instance instance : examples) {
      int labelIndex = this.labels.indexOf(instance.label);
      count.set(labelIndex, count.get(labelIndex).intValue() + 1);
    }

    for (Number value : count) {
      double valueRatio = value.doubleValue() / totalRows;
      classEntropy += valueRatio * Math.log(valueRatio) / Math.log(2);
    }
    return classEntropy * -1;
  }

  /**
   * Helper function to check if all examples have the same label.
   * 
   * @param instances
   * @return boolean
   */
  private boolean isAllExamplesHaveSameLabel(List<Instance> instances) {
    String headLabel = instances.get(0).label;
    for (Instance instance : instances) {
      if (!instance.label.equals(headLabel)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Find the attribute with the maximum information gain.
   * 
   * @param instances
   * @param attributes
   * @return String
   */
  private String findMaxImporantAttribute(List<Instance> instances, List<String> attributes) {
    double maxInfoGain = 0;

    String topAttribute = attributes.get(0);
    for (String attribute : attributes) {
      double infoGain = infoGain(attribute, instances);
      if (infoGain > maxInfoGain) {
        maxInfoGain = infoGain;
        topAttribute = attribute;
      }
    }

    if (maxInfoGain == 0) {
      List<String> sortedAttributes = new ArrayList<>(attributes);
      sortedAttributes.sort(String::compareTo);
      return sortedAttributes.get(0);
    }
    return topAttribute;
  }

  private DecTreeNode buildDecisionTree(List<Instance> examples, List<String> attributes) {
    return _buildDecisionTree(examples, attributes, examples, null);
  }

  private String getMostCommonLabel(List<Instance> examples) {
    Map<String, Number> outputMap = new HashMap<String, Number>();

    for (Instance instance : examples) {
      if (outputMap.containsKey(instance.label)) {
        outputMap.put(instance.label, outputMap.get(instance.label).intValue() + 1);
      } else {
        outputMap.put(instance.label, 1);
      }
    }

    String mostCommonOutput = "";
    int maxCount = 0;
    for (String key : outputMap.keySet()) {
      if (outputMap.get(key).intValue() > maxCount) {
        maxCount = outputMap.get(key).intValue();
        mostCommonOutput = key;
      }
    }

    return mostCommonOutput;
  }

  /**
   * Build a decision tree given a training set.
   * 
   * @param examples
   * @param attributes
   * @param parentExamples
   * @return DecTreeNode
   */
  private DecTreeNode _buildDecisionTree(List<Instance> examples, List<String> attributes,
      List<Instance> parentExamples, String parentAttribute) {
    if (examples.isEmpty()) {
      return new DecTreeNode(getMostCommonLabel(parentExamples), null, parentAttribute, true);
    } else if (isAllExamplesHaveSameLabel(examples)) {
      return new DecTreeNode(examples.get(0).label, null, parentAttribute, true);
    } else if (attributes.isEmpty()) {
      return new DecTreeNode(getMostCommonLabel(examples), null, parentAttribute, true);
    } else {
      String bestAttribute = findMaxImporantAttribute(examples, attributes);
      DecTreeNode node = new DecTreeNode(getMostCommonLabel(examples), bestAttribute, parentAttribute, false);
      List<String> allAttributeValues = attributeValues.get(bestAttribute);
      for (String value : allAttributeValues) {
        List<Instance> examplesWithAttributeValue = new ArrayList<Instance>();
        for (Instance instance : examples) {
          if (instance.attributes.get(getAttributeIndex(bestAttribute)).equals(value)) {
            examplesWithAttributeValue.add(instance);
          }
        }
        List<String> newAttributes = new ArrayList<String>(attributes);
        newAttributes.remove(bestAttribute);
        DecTreeNode child = _buildDecisionTree(examplesWithAttributeValue, newAttributes, examples, value);
        child.parentAttributeValue = value;
        node.addChild(child);
      }
      return node;
    }
  }
}
