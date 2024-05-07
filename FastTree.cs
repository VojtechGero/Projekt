using Microsoft.ML;
using Microsoft.ML.Trainers;
using Tensorflow.Contexts;

namespace Semestral;

internal class FastTree
{
    internal static List<SingleValueForecast> predict(List<Value> trainData, int scope)
    {
        var context = new MLContext();
        var data = context.Data.LoadFromEnumerable(trainData);
        var pipeline = context.Transforms.CopyColumns("Label", "zatizeniCerpani")
            .Append(context.Transforms.Concatenate("Features", "Year", "Month", "Day","Hour"))
            .Append(context.Regression.Trainers.FastTree("Label", "Features"));
        var model = pipeline.Fit(data);
        var predictionEngine = context.Model.CreatePredictionEngine<Value, SingleValueForecast>(model);
        var forecasts = new List<SingleValueForecast>();
        var last = trainData.Last();
        forecasts.Add( new SingleValueForecast(last.date,last.zatizeniCerpani));
        for (int i = 0; i < scope; i++)
        {
            var input = forecasts.Last();

            var forecast = predictionEngine.Predict(input.toValue());
            forecasts.Add(forecast);
        }

        return forecasts;
    }

}
