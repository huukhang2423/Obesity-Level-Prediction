import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import java.util.Random;

public class ObesityPredictionSVM {
    
    public static void main(String[] args) {
        try {
            // 1. Load dữ liệu
            System.out.println("=== LOADING DATA ===");
            String dataPath = "C:\\Users\\Admin\\Desktop\\Data minin proj\\data\\obesity_level.arff";
            DataSource source = new DataSource(dataPath);
            Instances data = source.getDataSet();
            
            // Kiểm tra dữ liệu
            System.out.println("Dataset: " + data.relationName());
            System.out.println("Number of instances: " + data.numInstances());
            System.out.println("Number of attributes: " + data.numAttributes());
            
            // In ra tên các attributes
            System.out.println("Attributes:");
            for (int i = 0; i < data.numAttributes(); i++) {
                System.out.println("  " + (i+1) + ": " + data.attribute(i).name());
            }
            
            // 2. Loại bỏ cột ID (attribute đầu tiên)
            System.out.println("\n=== PREPROCESSING ===");
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndices("1"); // Remove first attribute (ID)
            removeFilter.setInputFormat(data);
            data = Filter.useFilter(data, removeFilter);
            
            System.out.println("After removing ID - Number of attributes: " + data.numAttributes());
            
            // 3. Set class attribute (biến mục tiêu - attribute cuối cùng)
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
            System.out.println("Class attribute: " + data.classAttribute().name());
            System.out.println("Number of classes: " + data.numClasses());
            
            // In ra các class labels
            System.out.println("Class labels:");
            for (int i = 0; i < data.numClasses(); i++) {
                System.out.println("  " + i + ": " + data.classAttribute().value(i));
            }
            
            // 4. Normalize numeric attributes
            System.out.println("\n=== NORMALIZING DATA ===");
            Normalize normalize = new Normalize();
            normalize.setInputFormat(data);
            data = Filter.useFilter(data, normalize);
            System.out.println("Data normalized successfully");
            
            // 5. Tạo và cấu hình SVM (SMO) classifier
            System.out.println("\n=== CONFIGURING SVM CLASSIFIER ===");
            SMO svm = new SMO();
            
            // Cấu hình SVM parameters
            svm.setBuildLogisticModels(true); // For probability estimates
            svm.setC(1.0); // Complexity parameter (regularization)
            svm.setToleranceParameter(0.001); // Tolerance for stopping criterion
            
            // Sử dụng RBF Kernel (Radial Basis Function) - thường tốt hơn cho nhiều bài toán
            RBFKernel rbfKernel = new RBFKernel();
            rbfKernel.setGamma(0.01); // Gamma parameter for RBF kernel
            svm.setKernel(rbfKernel);
            
            System.out.println("SVM Configuration:");
            System.out.println("  C parameter: " + svm.getC());
            System.out.println("  Tolerance: " + svm.getToleranceParameter());
            System.out.println("  Kernel: " + svm.getKernel().getClass().getSimpleName());
            System.out.println("  RBF Gamma: " + rbfKernel.getGamma());
            
            // 6. Training và Evaluation với 10-fold Cross Validation
            System.out.println("\n=== TRAINING AND EVALUATION ===");
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(svm, data, 10, new Random(1));
            
            // 7. In kết quả chi tiết
            System.out.println("\n=== EVALUATION RESULTS ===");
            System.out.println("Correctly Classified Instances: " + 
                               (int)eval.correct() + " (" + 
                               String.format("%.4f", eval.pctCorrect()) + "%)");
            System.out.println("Incorrectly Classified Instances: " + 
                               (int)eval.incorrect() + " (" + 
                               String.format("%.4f", eval.pctIncorrect()) + "%)");
            
            System.out.println("\n=== DETAILED ACCURACY BY CLASS ===");
            System.out.println("Kappa Statistic: " + String.format("%.4f", eval.kappa()));
            System.out.println("Mean Absolute Error: " + String.format("%.4f", eval.meanAbsoluteError()));
            System.out.println("Root Mean Squared Error: " + String.format("%.4f", eval.rootMeanSquaredError()));
            
            // 8. Confusion Matrix
            System.out.println("\n=== CONFUSION MATRIX ===");
            double[][] confusionMatrix = eval.confusionMatrix();
            System.out.print("Actual\\Predicted\t");
            for (int i = 0; i < data.numClasses(); i++) {
                System.out.print(data.classAttribute().value(i) + "\t");
            }
            System.out.println();
            
            for (int i = 0; i < confusionMatrix.length; i++) {
                System.out.print(data.classAttribute().value(i) + "\t");
                for (int j = 0; j < confusionMatrix[i].length; j++) {
                    System.out.print((int)confusionMatrix[i][j] + "\t\t");
                }
                System.out.println();
            }
            
            // 9. Per Class Statistics
            System.out.println("\n=== PER CLASS DETAILED ACCURACY ===");
            System.out.println("Class\t\tPrecision\tRecall\t\tF-Measure");
            for (int i = 0; i < data.numClasses(); i++) {
                System.out.println(data.classAttribute().value(i) + "\t" +
                                 String.format("%.4f", eval.precision(i)) + "\t\t" +
                                 String.format("%.4f", eval.recall(i)) + "\t\t" +
                                 String.format("%.4f", eval.fMeasure(i)));
            }
            
            // 10. Training time
            System.out.println("\n=== ADDITIONAL INFO ===");
            System.out.println("Total Number of Instances: " + data.numInstances());
            System.out.println("Total Number of Features: " + (data.numAttributes() - 1));
            
            // 11. Thử với các kernel khác để so sánh
            System.out.println("\n=== COMPARING WITH POLYNOMIAL KERNEL ===");
            SMO svmPoly = new SMO();
            svmPoly.setBuildLogisticModels(true);
            svmPoly.setC(1.0);
            svmPoly.setToleranceParameter(0.001);
            
            PolyKernel polyKernel = new PolyKernel();
            polyKernel.setExponent(2.0); // Degree 2 polynomial
            svmPoly.setKernel(polyKernel);
            
            Evaluation evalPoly = new Evaluation(data);
            evalPoly.crossValidateModel(svmPoly, data, 10, new Random(1));
            
            System.out.println("Polynomial Kernel (degree 2) Accuracy: " + 
                             String.format("%.4f", evalPoly.pctCorrect()) + "%");
            System.out.println("RBF Kernel Accuracy: " + 
                             String.format("%.4f", eval.pctCorrect()) + "%");
            
            // 12. Save model (optional)
            System.out.println("\n=== MODEL TRAINING COMPLETED ===");
            System.out.println("Model ready for predictions!");
            
            // Để save model:
            // weka.core.SerializationHelper.writeAll("obesity_svm_model.model", svm, data);
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}