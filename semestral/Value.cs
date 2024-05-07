public class Value
{
    public DateTime date { get; set; }
    public float Year { get; set; }
    public float Month { get; set; }
    public float Day { get; set; }
    public float Hour { get; set; }
    public float zatizeniCerpani { get; set; }
    //[LoadColumn(2)]
    //public double zatizeni { get; set; }
    public Value(DateTime date, float zatizeniCerpani)
    {
        this.date = date;
        this.zatizeniCerpani = zatizeniCerpani;
        this.Year = date.Year;
        this.Month = date.Month;
        this.Day = date.Day;
        this.Hour = date.Hour;
    }
}

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