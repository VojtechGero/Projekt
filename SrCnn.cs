using Microsoft.ML;
using Microsoft.ML.TimeSeries;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Contexts;

namespace Semestral;

internal class SrCnn
{
    private static List<DoubleValue> toDouble(List<Value> values)
    {
        return values.Select(x => new DoubleValue(x.date, (double)x.zatizeniCerpani)).ToList();
    }

    internal static void detetectError(List<Value> trainData)
    {
        var doubleValues=toDouble(trainData);
        var context = new MLContext();

        var doubleData = context.Data.LoadFromEnumerable(doubleValues);
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
}
