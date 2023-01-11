package com.enpasos.djl.tests.djldocs.chapter3_linear_networks.softmax_regression_djl;

public class FashionMNIST {

    // Saved in the FashionMnist class for later use
    public static String[] getFashionMnistLabels(int[] labelIndices) {
        String[] textLabels = {"t-shirt", "trouser", "pullover", "dress", "coat",
            "sandal", "shirt", "sneaker", "bag", "ankle boot"};
        String[] convertedLabels = new String[labelIndices.length];
        for (int i = 0; i < labelIndices.length; i++) {
            convertedLabels[i] = textLabels[labelIndices[i]];
        }
        return convertedLabels;
    }

    public static String getFashionMnistLabel(int labelIndice) {
        String[] textLabels = {"t-shirt", "trouser", "pullover", "dress", "coat",
            "sandal", "shirt", "sneaker", "bag", "ankle boot"};
        return textLabels[labelIndice];
    }
}
