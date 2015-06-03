package org.deeplearning4j.deepbelief;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;

/**
 * Created by agibsonccc on 9/11/14.
 */
public class DBNFullMnistExample {

    private static Logger log = LoggerFactory.getLogger(DBNFullMnistExample.class);

    public static void main(String[] args) throws Exception {
        Nd4j.dtype = DataBuffer.Type.FLOAT;

        log.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(500,5000);
        
        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(784)
                .nOut(10)
                .activationFunction("sigmoid")
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0, 1))
                .constrainGradientToUnitNorm(true)
                .iterations(5)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f)
                .momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(4)
                .hiddenLayerSizes(new int[]{400, 250, 100})
                .override(3, new ClassifierOverride())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        Collections.singletonList((IterationListener) new ScoreIterationListener(1));

        log.info("Train model....");
        model.fit(iter); // achieves end to end pre-training

        // model.fit(iter) // alternate approach that does end-to-end training before fine tuning

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();

        DataSetIterator testIter = new MnistDataSetIterator(100,2000);
        while(testIter.hasNext()) {
            DataSet testMnist = testIter.next();
            testMnist.normalizeZeroMeanZeroUnitVariance();
            INDArray predict2 = model.output(testMnist.getFeatureMatrix());
            eval.eval(testMnist.getLabels(), predict2);
        }

        log.info(eval.stats());
        log.info("Accuracy: " + eval.accuracy());
        log.info("****************Example finished********************");

    }

}
