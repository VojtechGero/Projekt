using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Semestral;

internal static class ForecastSsa
{
    internal static List<float> forecast(List<Value> trainData,int scope)
    {
        var context = new MLContext();
        var data = context.Data.LoadFromEnumerable(trainData);
        
        var pipeline = context.Forecasting.ForecastBySsa(
            nameof(ValueForcasts.Forecasts),
            nameof(Value.zatizeniCerpani),
            windowSize: 24 * 7,
            seriesLength: 24 * 30,
            trainSize: trainData.Count,
            horizon: scope,
            confidenceLevel: 0.95f,
            confidenceLowerBoundColumn: "LowerBoundRentals",
            confidenceUpperBoundColumn: "UpperBoundRentals"
            );
        var model = pipeline.Fit(data);
        var forecastingEngine = model.CreateTimeSeriesEngine<Value, ValueForcasts>(context);
        var forecast = forecastingEngine.Predict();
        return forecast.Forecasts.ToList();
    }
}
