package com.enpasos.djl.tests.common;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

@Data
public class PerformanceResult {
    private List<DurAndMem> durations = new ArrayList<>();

    public void add(DurAndMem duration) {
        durations.add(duration);
    }

    public void logResultInExcelFriendlyFormatToConsole() {
        System.out.println("epoch;duration[ms];gpuMem[MiB]");
        IntStream.range(0, durations.size()).forEach(i -> System.out.println(i + ";" + durations.get(i).getDur() + ";" + durations.get(i).getMem() / 1024 / 1024));
    }

    public DurAndMem getFirstDurAndMem() {
        return durations.get(0);
    }

    public DurAndMem getLastDurAndMem() {
        return durations.get(durations.size() - 1);
    }


    public int numOfMeasurements() {
        return durations.size();
    }

    public int durationIncrease() {
        return (int) (getLastDurAndMem().getDur() - getFirstDurAndMem().getDur());
    }

    public int memoryIncrease() {
        return (int) (getLastDurAndMem().getMem() - getFirstDurAndMem().getMem());
    }

    public double relativeDurationIncrease() {
        return (double) durationIncrease() / getFirstDurAndMem().getDur();
    }

    public double relativeMemoryIncrease() {
        return (double) memoryIncrease() / getFirstDurAndMem().getMem();
    }


}
