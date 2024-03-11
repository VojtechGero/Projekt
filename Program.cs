using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using Semestral;

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
    windowSize: 24*3,
    seriesLength: 24*7,
    trainSize: trainData.Count,
    horizon: testData.Count
    );
var model = pipeline.Fit(data);
var forecastingEngine=model.CreateTimeSeriesEngine<Value,ValueForcast>(context);
var forecast = forecastingEngine.Predict();

Utilities.evaluate(testData, forecast.Forecast.ToList());


internal class ValueForcast
{
    public float[] Forecast {  get; set; }
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