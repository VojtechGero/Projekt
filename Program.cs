using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TimeSeries;
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
List<DoubleValue> doubleCon = trainData.Select(x => new DoubleValue(x)).ToList();
var doubleData=context.Data.LoadFromEnumerable(doubleCon);
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
int period = context.AnomalyDetection.DetectSeasonality(
        doubleData, 
        nameof(DoubleValue.zatizeniCerpani),
        seasonalityWindowSize: 400);

/*
Threshold = 0.2,
Sensitivity = 76,
citlivější na menší změny
a chytne jen tu změnu

Threshold = 0.35,
pak chytá aj ty v dalších dnech
*/
var options = new SrCnnEntireAnomalyDetectorOptions()
{
    Threshold = 0.35,
    Sensitivity = 76,
    DetectMode = SrCnnDetectMode.AnomalyAndMargin,
    Period = period,
};
var outputDataView = context.AnomalyDetection.DetectEntireAnomalyBySrCnn(
        doubleData, nameof(ValuePrediction.Prediction), nameof(DoubleValue.zatizeniCerpani), options);

int index = 0;
bool e = false;
foreach (var i in context.Data.CreateEnumerable<ValuePrediction>(outputDataView, reuseRowObject: false))
{
    ++index;
    if (i.Prediction[0] == 1)
    {
        Console.WriteLine($"Anomaly detected at {trainData[index].date}");
        e = true;
    }
}
if (!e)
{
    Console.WriteLine("No error found");
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

internal class DoubleValue
{
    public DateTime date { get; set; }
    public double zatizeniCerpani { get; set; }
    //[LoadColumn(2)]
    //public double zatizeni { get; set; }
    public DoubleValue(DateTime date, double zatizeniCerpani)
    {
        this.date = date;
        this.zatizeniCerpani = zatizeniCerpani;
    }
    public DoubleValue(Value v)
    {
        this.date = v.date;
        this.zatizeniCerpani = v.zatizeniCerpani;
    }
}