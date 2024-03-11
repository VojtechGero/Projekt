

/*
DateOnly getDate(string date)
{
    string[] parts = date.Split(".");
    DateOnly dateOnly = new DateOnly(int.Parse(parts[2]), int.Parse(parts[1]), int.Parse(parts[0]));
    return dateOnly;
}

double[] aggregate(double[] values,int num)
{
    var doubles=new List<double>();
    int overhang = values.Length % num;
    double sum;
    for (int i = 0; i < values.Length - overhang; i += num)
    {
        sum = 0;
        for(int j = i; j < i + num; j++)
        {
            sum += values[j];
        }
        doubles.Add(sum/num);
    }
    sum = 0;
    for (int i = values.Length - overhang;i< values.Length; i++)
    {
        sum+= values[i];
    }
    doubles.Add(sum / overhang);
    return doubles.ToArray();
}

string path = "C:\\Users\\vojte\\OneDrive\\Plocha\\data.csv";
string[] lines=File.ReadAllLines(path);
List<Value> values = new List<Value>();
for(int i=3; i<lines.Length; i++)
{
    string[] parts = lines[i].Split(";",options:StringSplitOptions.RemoveEmptyEntries);
    values.Add(new Value(getDate(parts[0]), double.Parse(parts[1]), double.Parse(parts[2])));
}
double sum = 0;
foreach(Value value in values)
{
    sum += value.zatizeni;
    Console.WriteLine($"{value.zatizeni}");
}
Console.WriteLine($"\nPrůměr: {sum/values.Count}\n");
sum = 0;
double[] d = aggregate(values.Select(x => x.zatizeni).ToArray(), 7);
foreach (var i in d)
{
    sum += i;
    Console.WriteLine(i);
}
Console.WriteLine("\n"+sum / d.Length);

class Value
{
    public DateOnly date { get; set; }
    public double zatizeniCerpani { get; set; }
    public double zatizeni {  get; set; }

    public Value(DateOnly date, double zatizeniCerpani, double zatizeni)
    {
        this.date = date;
        this.zatizeniCerpani = zatizeniCerpani;
        this.zatizeni = zatizeni;
    }
}
*/



using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using Semestral;



string trainPath = "..\\..\\..\\train.txt";
string testPath = "..\\..\\..\\test.txt";

var context=new MLContext();
var data = context.Data.LoadFromEnumerable(Utilities.getData(trainPath));
List<Value> test = Utilities.getData(testPath);
var pipeline = context.Forecasting.ForecastBySsa(
    "Forecast",
    nameof(Value.zatizeniCerpani),
    windowSize: 24*7,
    seriesLength: 24*30,
    trainSize: File.ReadAllLines(trainPath).Length - 1,
    horizon: File.ReadAllLines(testPath).Length - 1
    );
var model = pipeline.Fit(data);
var forecastingEngine=model.CreateTimeSeriesEngine<Value,ValueForcast>(context);
var forecast = forecastingEngine.Predict();

Utilities.evaluate(test, forecast.Forecast.ToList());


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