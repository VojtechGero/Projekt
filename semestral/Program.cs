using Microsoft.ML;
using Microsoft.ML.TimeSeries;
using Microsoft.ML.Transforms.TimeSeries;
using Semestral;
using System.Data;
using Tensorflow.Contexts;

List<Value> getData(string filePath)
{
    List<Value> data = new List<Value>();
    string[] lines = File.ReadAllLines(filePath);
    foreach (string line in lines)
    {
        if (!char.IsLetter(line[0]))
        {
            string[] vals = line.Split(';');
            DateTime datum = DateTime.Parse(vals[0]);
            float zatizeniSCerpanim = float.Parse(vals[1]);
            data.Add(new Value(datum, zatizeniSCerpanim));
        }
    }
    return data;
}

string trainPath = "..\\..\\..\\data\\train.txt";
string testPath = "..\\..\\..\\data\\test.txt";

List<Value> trainData = getData(trainPath);
List<Value> testData = getData(testPath);
var reg = new Regression();
Utilities.evaluate(testData, reg.FastTree(trainData, testData.Count), "FastTree");
Utilities.evaluate(testData, reg.LbfgsPoisson(trainData, testData.Count), "LbfgsPoisson");
Utilities.evaluate(testData, reg.FastForest(trainData, testData.Count), "FastForest");
Utilities.evaluate(testData, reg.Sdca(trainData, testData.Count), "Sdca");
Utilities.evaluate(testData, reg.Gam(trainData, testData.Count), "Gam");
Utilities.evaluate(testData, ForecastSsa.forecast(trainData, testData.Count), "ForecastBySsa");

SrCnn.detetectError(trainData);

/*
var trainigPipeLine = context.Transforms.DetectSpikeBySsa(
                "Prediction",
                "zatizeniCerpani",
                confidence: 98.0,
                pvalueHistoryLength: 30,
                trainingWindowSize: 90,
                seasonalityWindowSize: 30);

ITransformer trainedModel = trainigPipeLine.Fit(data);
var transformedData = trainedModel.Transform(data);
int index = 0;
foreach (var i in context.Data.CreateEnumerable<ValuePrediction>(transformedData, reuseRowObject: false))
{
    if (i.Prediction[0] == 1)
    {
        Console.WriteLine($"Anomaly detected at {trainData[index].date}");
    }
    ++index;
}
*/