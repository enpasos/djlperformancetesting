package com.enpasos.djl.tests.djldocs.chapter4_multilayer_perceptrons.mlp_djl;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
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
public final class App4 implements App {


    public String source() {
        return "https://d2l.djl.ai/chapter_multilayer-perceptrons/mlp-djl.html";
    }


    public PerformanceResult run() {

        PerformanceResult pr = new PerformanceResult();

        int batchSize = 256;
        int numEpochs = Integer.getInteger("MAX_EPOCH", 10);


        boolean randomShuffle = true;

        //  try (NDManager manager = NDManager.newBaseManager();) {

        double[] trainLoss;
        double[] testAccuracy;
        double[] epochCount;
        double[] trainAccuracy;

        trainLoss = new double[numEpochs];
        trainAccuracy = new double[numEpochs];
        testAccuracy = new double[numEpochs];
        epochCount = new double[numEpochs];

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


        for (int i = 0; i < epochCount.length; i++) {
            epochCount[i] = (i + 1);
        }

        Map<String, double[]> evaluatorMetrics = new HashMap<>();

        SequentialBlock net = new SequentialBlock();
        net.add(Blocks.batchFlattenBlock(784));
        net.add(Linear.builder().setUnits(256).build());
        net.add(Activation::relu);
        net.add(Linear.builder().setUnits(10).build());
        net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);

        Tracker lrt = Tracker.fixed(0.5f);
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

        Loss loss = Loss.softmaxCrossEntropyLoss();

        DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
            .optOptimizer(sgd) // Optimizer (loss function)
            .optDevices(Engine.getInstance().getDevices(1)) // single GPU
            .addEvaluator(new Accuracy()) // Model Accuracy
            .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging


        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(net);

            try (Trainer trainer = model.newTrainer(config)) {

                trainer.initialize(new Shape(1, 784));
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

            }

        }
        return pr;
    }
}




