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
    this.root = _buildDecisionTree(train.instances, train.attributes, null);
  }

  @Override
  public String classify(Instance instance) {
    DecTreeNode currentNode = root;
    while (!currentNode.terminal) {
      String attribute = currentNode.attribute;
      String value = instance.attributes.get(getAttributeIndex(attribute));
      int valueIndex = getAttributeValueIndex(attribute, value);
      currentNode = currentNode.children.get(valueIndex);
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
    System.out.format("%.5f\n", this.getAccuracy(test));
  }

  /**
   * Get the accuracy of the decision tree on a given test set.
   * 
   * @param test: the test set
   * @return the accuracy
   */
  private double getAccuracy(DataSet test) {
    int correct = 0;
    for (Instance instance : test.instances) {
      if (classify(instance).equals(instance.label)) {
        correct++;
      }
    }
    return (double) correct / test.instances.size();
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

    this.root = _buildDecisionTree(train.instances, train.attributes, null);
    pruneTree(this.root, tune);
  }

  /**
   * Prune the decision tree using the given tuning set.
   * @param node the current node being pruned
   * @param tune
   */
  private void pruneTree(DecTreeNode node, DataSet tune) {
    if (node.terminal) {
      return;
    }

    for (DecTreeNode child : node.children) {
      this.pruneTree(child, tune);
    }

    List<DecTreeNode> originalChildren = new ArrayList<DecTreeNode>(node.children);
    String originalLabel = node.label;

    double accuracyWithoutPruning = getAccuracy(tune);

    node.label = getMostCommonLabel(node);
    node.terminal = true;
    node.children = null;

    double accuracyWithPruning = getAccuracy(tune);

    // If accuracy is better or the same without pruning, revert.
    if (accuracyWithPruning < accuracyWithoutPruning) {
      // Revert pruning.
      node.terminal = false;
      node.label = originalLabel;
      node.children = originalChildren;
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
   * Count how many rows have the given attributeValue for the given attribute for each attributeValue
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
   * Helper function to get the most common output.
   * 
   * @param examples
   * @return DecTreeNode
   */
  private DecTreeNode getMostCommonOutput(List<Instance> examples) {
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
    return new DecTreeNode(mostCommonOutput, null, null, true);
  }

  /**
   * Helper function to get the most common label.
   * 
   * @param node
   * @return String
   */
  private String getMostCommonLabel(DecTreeNode node) {
    Map<String, Number> labelMap = new HashMap<String, Number>();
    for (DecTreeNode child : node.children) {
      if (child.label == null) {
        continue;
      }
      if (labelMap.containsKey(child.label)) {
        labelMap.put(child.label, labelMap.get(child.label).intValue() + 1);
      } else {
        labelMap.put(child.label, 1);
      }
    }

    String mostCommonOutput = "";
    int maxCount = 0;
    for (String key : labelMap.keySet()) {
      if (labelMap.get(key).intValue() > maxCount) {
        maxCount = labelMap.get(key).intValue();
        mostCommonOutput = key;
      }
    }

    return mostCommonOutput;
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
    return topAttribute;
  }

  // instance: G[x, b, c, o, h, c, 1, 2, y, y]
  /**
   * Build a decision tree given a training set.
   * 
   * @param examples
   * @param attributes
   * @param parentExamples
   * @return DecTreeNode
   */
  private DecTreeNode _buildDecisionTree(List<Instance> examples, List<String> attributes,
      List<Instance> parentExamples) {
    if (examples.isEmpty()) {
      return getMostCommonOutput(parentExamples);
    } else if (isAllExamplesHaveSameLabel(examples)) {
      return new DecTreeNode(examples.get(0).label, null, null, true);
    } else if (attributes.isEmpty()) {
      return getMostCommonOutput(examples);
    } else {
      String bestAttribute = findMaxImporantAttribute(examples, attributes);
      DecTreeNode node = new DecTreeNode(null, bestAttribute, null, false);
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
        DecTreeNode child = _buildDecisionTree(examplesWithAttributeValue, newAttributes, examples);
        child.parentAttributeValue = value;
        node.addChild(child);
      }
      return node;
    }
  }
}
