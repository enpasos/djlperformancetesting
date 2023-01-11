package com.enpasos.djl.tests.djldocs.chapter3_linear_networks.softmax_regression_djl;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
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


@Slf4j
public final class App3 implements App {


    public String source() {
        return "https://d2l.djl.ai/chapter_linear-networks/softmax-regression-djl.html";
    }


    public PerformanceResult run() {

        PerformanceResult pr = new PerformanceResult();

        int batchSize = 256;
        int numEpochs = 100;


        boolean randomShuffle = true;

        try (NDManager manager = NDManager.newBaseManager();) {

            FashionMnist trainingSet = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

            FashionMnist validationSet = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, false)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();


            try (Model model = Model.newInstance("softmax-regression")) {


                SequentialBlock net = new SequentialBlock();
                net.add(Blocks.batchFlattenBlock(28 * 28)); // flatten input
                net.add(Linear.builder().setUnits(10).build()); // set 10 output channels

                model.setBlock(net);


                Loss loss = Loss.softmaxCrossEntropyLoss();

                Tracker lrt = Tracker.fixed(0.1f);
                Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

                DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                    .optOptimizer(sgd) // Optimizer
                    .optDevices(manager.getEngine().getDevices(1)) // single GPU
                    .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

                Shape shape = new Shape(1, 28 * 28);

                // the following code is identical to reproducebug3 - where it seems to work
                for (int epoch = 1; epoch <= numEpochs; epoch++) {

                    try (Trainer trainer = model.newTrainer(config)) {

                        trainer.initialize(shape);
                        Metrics metrics = new Metrics();
                        trainer.setMetrics(metrics);

                        DurAndMem duration = new DurAndMem();
                        duration.on();
                        System.out.printf("Epoch %d\n", epoch);

                        try {
                            for (Batch batch : trainer.iterateDataset(trainingSet)) {
                                EasyTrain.trainBatch(trainer, batch);
                                trainer.step();
                                batch.close();
                            }
                           EasyTrain.evaluateDataset(trainer, validationSet);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        } catch (TranslateException e) {
                            throw new RuntimeException(e);
                        }
                        trainer.notifyListeners(listener -> listener.onEpoch(trainer));

                        duration.off();
                        pr.add(duration);

                    }

                }
             }
        }
        return pr;
    }
}




