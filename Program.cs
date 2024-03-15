using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using Semestral;
using System.Data;
using System.Numerics;

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


string trainPath = "..\\..\\..\\train.txt";
string testPath = "..\\..\\..\\test.txt";

var context=new MLContext();
List<Value> trainData = getData(trainPath);
var data = context.Data.LoadFromEnumerable(trainData);
List<Value> testData = getData(testPath);
var pipeline = context.Forecasting.ForecastBySsa(
    "Forecast",
    nameof(Value.zatizeniCerpani),
    windowSize: 24*7,
    seriesLength: 24*30,
    trainSize: trainData.Count,
    horizon: testData.Count,
    confidenceLevel: 0.95f,
    confidenceLowerBoundColumn: "LowerBoundRentals",
    confidenceUpperBoundColumn: "UpperBoundRentals"
    );
var model = pipeline.Fit(data);
var forecastingEngine=model.CreateTimeSeriesEngine<Value,ValueForcast>(context);
var forecast = forecastingEngine.Predict();

Utilities.evaluate(testData, forecast.Forecast.ToList());

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

internal class ValueForcast
{
    public float[] Forecast {  get; set; }
}

internal class ValuePrediction
{
    public double[] Prediction { get; set; }
}
internal class Value
{
    public DateTime date { get; set; }
    public float zatizeniCerpani { get; set; }
    //[LoadColumn(2)]
    //public double zatizeni { get; set; }
    public Value(DateTime date, float zatizeniCerpani)
    {
        this.date = date;
        this.zatizeniCerpani = zatizeniCerpani;
    }
}