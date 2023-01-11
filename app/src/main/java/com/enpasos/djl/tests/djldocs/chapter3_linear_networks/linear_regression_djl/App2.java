package com.enpasos.djl.tests.djldocs.chapter3_linear_networks.linear_regression_djl;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import com.enpasos.djl.tests.common.App;
import com.enpasos.djl.tests.common.DurAndMem;
import com.enpasos.djl.tests.common.PerformanceResult;
import com.enpasos.djl.tests.djldocs.chapter3_linear_networks.DataPoints;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;

import static com.enpasos.djl.tests.djldocs.chapter3_linear_networks.linear_regression_djl.Utils.loadArray;


@Slf4j
public final class App2 implements App {


    public String source() {
        return "https://d2l.djl.ai/chapter_linear-networks/linear-regression-djl.html";
    }


    public PerformanceResult run() {

        PerformanceResult pr = new PerformanceResult();
        try (NDManager manager = NDManager.newBaseManager();) {

            NDArray trueW = manager.create(new float[]{2, -3.4f});
            float trueB = 4.2f;

            DataPoints dp = syntheticData(manager, trueW, trueB, 100000);
            NDArray features = dp.getX();
            NDArray labels = dp.getY();


            int batchSize = 10;
            ArrayDataset dataset = loadArray(features, labels, batchSize, false);


            try (Model model = Model.newInstance("lin-reg")) {

                SequentialBlock net = new SequentialBlock();
                Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build();
                net.add(linearBlock);

                model.setBlock(net);

                Loss l2loss = Loss.l2Loss();

                Tracker lrt = Tracker.fixed(0.03f);
                Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();
                DefaultTrainingConfig config = new DefaultTrainingConfig(l2loss)
                    .optOptimizer(sgd) // Optimizer (loss function)
                    .optDevices(manager.getEngine().getDevices(1)) // single GPU
                    .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

                int numEpochs = 10;

                Shape shape = new Shape(batchSize, 2);

                try (Trainer trainer = model.newTrainer(config)) {

                    trainer.initialize(shape);
                    Metrics metrics = new Metrics();
                    trainer.setMetrics(metrics);

                    for (int epoch = 1; epoch <= numEpochs; epoch++) {

                        DurAndMem duration = new DurAndMem();
                        duration.on();

                        try {
                            for (Batch batch : trainer.iterateDataset(dataset)) {
                                EasyTrain.trainBatch(trainer, batch);
                                trainer.step();
                                batch.close();
                            }
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


    // Generate y = X w + b + noise
    public static DataPoints syntheticData(NDManager manager, NDArray w, float b, int numExamples) {
        NDArray X = manager.randomNormal(new Shape(numExamples, w.size()));
        NDArray y = X.matMul(w.transpose()).add(b);
        // Add noise
        y = y.add(manager.randomNormal(0, 0.01f, y.getShape(), DataType.FLOAT32));
        return new DataPoints(X, y);
    }

}
