package com.enpasos.djl.tests;

import ai.djl.ndarray.gc.SwitchGarbageCollection;
import com.enpasos.djl.tests.common.App;
import com.enpasos.djl.tests.common.PerformanceResult;
import com.enpasos.djl.tests.djldocs.chapter6_convolutional_neural_networks.lenet.App5;
import com.enpasos.djl.tests.djldocs.chapter3_linear_networks.linear_regression_djl.App2;
import com.enpasos.djl.tests.djldocs.chapter3_linear_networks.linear_regression_scratch.App1;
import com.enpasos.djl.tests.djldocs.chapter3_linear_networks.softmax_regression_djl.App3;
import com.enpasos.djl.tests.djldocs.chapter4_multilayer_perceptrons.mlp_djl.App4;
import com.enpasos.djl.tests.djldocs.chapter7_convolutional_modern.alexnet.App6;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
class Tests {

    @BeforeAll
    static void switchGarbageCollectionOn() {
        SwitchGarbageCollection.on();
    }

    @Test
    void runApp1() {
        App app = new App1();
        PerformanceResult pr = app.run();
        log.info("Source: {}", app.source());
        pr.logResultInExcelFriendlyFormatToConsole();

        assertTrue(pr.memoryIncrease() <= 2*1024*1024);  // accept an initial jump of 2 MB
        assertTrue(pr.relativeDurationIncrease() <= 0.01);  // accept an increase of 1 %
    }

    @Test
    void runApp2() {
        App app = new App2();
        PerformanceResult pr = app.run();
        log.info("Source: {}", app.source());
        pr.logResultInExcelFriendlyFormatToConsole();

        assertTrue(pr.memoryIncrease() <= 0);  // accept no memory increase
        assertTrue(pr.relativeDurationIncrease() <= 0.001);  // accept an increase of 0.1 %
    }


    @Test
    void runApp3() {
        App app = new App3();
        PerformanceResult pr = app.run();
        log.info("Source: {}", app.source());
        pr.logResultInExcelFriendlyFormatToConsole();

        assertTrue(pr.memoryIncrease() <= 0);  // accept no memory increase
        assertTrue(pr.relativeDurationIncrease() <= 0.001);  // accept an increase of 0.1 %
    }


    @Test
    void runApp4() {
        App app = new App4();
        PerformanceResult pr = app.run();
        log.info("Source: {}", app.source());
        pr.logResultInExcelFriendlyFormatToConsole();

        assertTrue(pr.memoryIncrease() <= 0);  // accept no memory increase
        assertTrue(pr.relativeDurationIncrease() <= 0.001);  // accept an increase of 0.1 %
    }

    @Test
    void runApp5() {
        App app = new App5();
        PerformanceResult pr = app.run();
        log.info("Source: {}", app.source());
        pr.logResultInExcelFriendlyFormatToConsole();

        assertTrue(pr.memoryIncrease() <= 0);  // accept no memory increase
        assertTrue(pr.relativeDurationIncrease() <= 0.001);  // accept an increase of 0.1 %
    }


//    @Test
//    void runApp6() {
//        App app = new App6();
//        PerformanceResult pr = app.run();
//        log.info("Source: {}", app.source());
//        pr.logResultInExcelFriendlyFormatToConsole();
//
//        assertTrue(pr.memoryIncrease() <= 0);  // accept no memory increase
//        assertTrue(pr.relativeDurationIncrease() <= 0.05);  // accept an increase of 5 %
//    }

}
