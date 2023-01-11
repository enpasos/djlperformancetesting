package com.enpasos.djl.tests.djldocs.chapter6_convolutional_neural_networks.lenet;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import com.enpasos.djl.tests.common.App;
import com.enpasos.djl.tests.common.DurAndMem;
import com.enpasos.djl.tests.common.PerformanceResult;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


@Slf4j
public final class App5 implements App {


    public String source() {
        return "https://d2l.djl.ai/chapter_convolutional-neural-networks/lenet.html";
    }


    public PerformanceResult run() {

        PerformanceResult pr = new PerformanceResult();

        Engine.getInstance().setRandomSeed(1111);

        try (NDManager manager = NDManager.newBaseManager()) {
            SequentialBlock block = new SequentialBlock();
            block
                .add(Conv2d.builder()
                    .setKernelShape(new Shape(5, 5))
                    .optPadding(new Shape(2, 2))
                    .optBias(false)
                    .setFilters(6)
                    .build())
                .add(Activation::sigmoid)
                .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
                .add(Conv2d.builder()
                    .setKernelShape(new Shape(5, 5))
                    .setFilters(16).build())
                .add(Activation::sigmoid)
                .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
                // Blocks.batchFlattenBlock() will transform the input of the shape (batch size, channel,
                // height, width) into the input of the shape (batch size,
                // channel * height * width)
                .add(Blocks.batchFlattenBlock())
                .add(Linear
                    .builder()
                    .setUnits(120)
                    .build())
                .add(Activation::sigmoid)
                .add(Linear
                    .builder()
                    .setUnits(84)
                    .build())
                .add(Activation::sigmoid)
                .add(Linear
                    .builder()
                    .setUnits(10)
                    .build());

            float lr = 0.9f;


            int batchSize = 256;
            int numEpochs = Integer.getInteger("MAX_EPOCH", 100);
            double[] trainLoss;
            double[] testAccuracy;
            double[] epochCount;
            double[] trainAccuracy;

            epochCount = new double[numEpochs];

            for (int i = 0; i < epochCount.length; i++) {
                epochCount[i] = (i + 1);
            }

            FashionMnist trainIter = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, true)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();


            FashionMnist testIter = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, true)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

            try {
                trainIter.prepare();
                testIter.prepare();
            } catch (IOException e) {
                throw new RuntimeException(e);
            } catch (TranslateException e) {
                throw new RuntimeException(e);
            }


            try (Model model = Model.newInstance("cnn")) {
                model.setBlock(block);

                Loss loss = Loss.softmaxCrossEntropyLoss();

                Tracker lrt = Tracker.fixed(lr);
                Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

                DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
                    .optDevices(Engine.getInstance().getDevices(1)) // Single GPU
                    .addEvaluator(new Accuracy()) // Model Accuracy
                    .addTrainingListeners(TrainingListener.Defaults.basic());

                try (Trainer trainer = model.newTrainer(config)) {

                    NDArray X = manager.randomUniform(0f, 1.0f, new Shape(1, 1, 28, 28));
                    trainer.initialize(X.getShape());

                    Shape currentShape = X.getShape();


                    for (int i = 0; i < block.getChildren().size(); i++) {
                        Shape[] newShape = block.getChildren().get(i).getValue().getOutputShapes(new Shape[]{currentShape});
                        currentShape = newShape[0];
                        System.out.println(block.getChildren().get(i).getKey() + " layer output : " + currentShape);
                    }


                    double avgTrainTimePerEpoch = 0;
                    Map<String, double[]> evaluatorMetrics = new HashMap<>();

                    trainer.setMetrics(new Metrics());


                    // the following code is identical to reproducebug3 - where it seems to work
                    for (int epoch = 1; epoch <= numEpochs; epoch++) {

                        DurAndMem duration = new DurAndMem();
                        duration.on();
                        //System.out.printf("Epoch %d\n", epoch);

                        try {
                            for (Batch batch : trainer.iterateDataset(trainIter)) {
                                EasyTrain.trainBatch(trainer, batch);
                                trainer.step();
                                batch.close();
                            }
                            EasyTrain.evaluateDataset(trainer, testIter);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        } catch (TranslateException e) {
                            throw new RuntimeException(e);
                        }
                        trainer.notifyListeners(listener -> listener.onEpoch(trainer));

                        duration.off();
                        pr.add(duration);

                    }
                    Metrics metrics = trainer.getMetrics();

                    trainer.getEvaluators().stream()
                        .forEach(evaluator -> {
                            evaluatorMetrics.put("train_epoch_" + evaluator.getName(), metrics.getMetric("train_epoch_" + evaluator.getName()).stream()
                                .mapToDouble(x -> x.getValue().doubleValue()).toArray());
                            evaluatorMetrics.put("validate_epoch_" + evaluator.getName(), metrics.getMetric("validate_epoch_" + evaluator.getName()).stream()
                                .mapToDouble(x -> x.getValue().doubleValue()).toArray());
                        });

                    avgTrainTimePerEpoch = metrics.mean("epoch");

                    trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
                    trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
                    testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

                    System.out.printf("loss %.3f,", trainLoss[numEpochs - 1]);
                    System.out.printf(" train acc %.3f,", trainAccuracy[numEpochs - 1]);
                    System.out.printf(" test acc %.3f\n", testAccuracy[numEpochs - 1]);
                    System.out.printf("%.1f examples/sec \n", trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10, 9)));

                }

            }
        }
        return pr;
    }
}




