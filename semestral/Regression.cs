using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Contexts;

namespace Semestral;

public class Regression
{
    MLContext context { get; set; }
    public Regression()
    {
        context= new MLContext();
    }
    
    public List<float> FastTree(List<Value> trainData, int scope)
    {
        var data = context.Data.LoadFromEnumerable(trainData);
        var pipeline = context.Transforms.CopyColumns("Label", "zatizeniCerpani")
            .Append(context.Transforms.Concatenate("Features", "Year", "Month", "Day", "Hour"))
            .Append(context.Regression.Trainers.FastTree("Label", "Features"));

        var model = pipeline.Fit(data);
        var predictionEngine = context.Model.CreatePredictionEngine<Value, SingleValueForecast>(model);
        var forecasts = new List<SingleValueForecast>();

        var last = trainData.Last();
        var nextDate = last.date;

        for (int i = 0; i < scope; i++)
        {
            nextDate = nextDate.AddHours(1);
            var nextValue = new Value(nextDate, 0);
            var forecast = predictionEngine.Predict(nextValue);
            forecasts.Add(new SingleValueForecast
            {
                Year = nextValue.Year,
                Month = nextValue.Month,
                Day = nextValue.Day,
                Hour = nextValue.Hour,
                zatizeniCerpani = forecast.PredictedZatizeniCerpani,
                PredictedZatizeniCerpani = forecast.PredictedZatizeniCerpani
            });
        }
        return forecasts.Select(x=>x.PredictedZatizeniCerpani).ToList();
    }
    public List<float> LbfgsPoisson(List<Value> trainData, int scope)
    {
        var data = context.Data.LoadFromEnumerable(trainData);
        var pipeline = context.Transforms.CopyColumns("Label", "zatizeniCerpani")
            .Append(context.Transforms.Concatenate("Features", "Year", "Month", "Day", "Hour"))
            .Append(context.Regression.Trainers.LbfgsPoissonRegression("Label", "Features"));

        var model = pipeline.Fit(data);
        var predictionEngine = context.Model.CreatePredictionEngine<Value, SingleValueForecast>(model);
        var forecasts = new List<SingleValueForecast>();

        var last = trainData.Last();
        var nextDate = last.date;

        for (int i = 0; i < scope; i++)
        {
            nextDate = nextDate.AddHours(1);
            var nextValue = new Value(nextDate, 0);
            var forecast = predictionEngine.Predict(nextValue);
            forecasts.Add(new SingleValueForecast
            {
                Year = nextValue.Year,
                Month = nextValue.Month,
                Day = nextValue.Day,
                Hour = nextValue.Hour,
                zatizeniCerpani = forecast.PredictedZatizeniCerpani,
                PredictedZatizeniCerpani = forecast.PredictedZatizeniCerpani
            });
        }
        return forecasts.Select(x => x.PredictedZatizeniCerpani).ToList();
    }

    public List<float> FastForest(List<Value> trainData, int scope)
    {
        var data = context.Data.LoadFromEnumerable(trainData);
        var pipeline = context.Transforms.CopyColumns("Label", "zatizeniCerpani")
            .Append(context.Transforms.Concatenate("Features", "Year", "Month", "Day", "Hour"))
            .Append(context.Regression.Trainers.FastForest("Label", "Features"));

        var model = pipeline.Fit(data);
        var predictionEngine = context.Model.CreatePredictionEngine<Value, SingleValueForecast>(model);
        var forecasts = new List<SingleValueForecast>();

        var last = trainData.Last();
        var nextDate = last.date;

        for (int i = 0; i < scope; i++)
        {
            nextDate = nextDate.AddHours(1);
            var nextValue = new Value(nextDate, 0);
            var forecast = predictionEngine.Predict(nextValue);
            forecasts.Add(new SingleValueForecast
            {
                Year = nextValue.Year,
                Month = nextValue.Month,
                Day = nextValue.Day,
                Hour = nextValue.Hour,
                zatizeniCerpani = forecast.PredictedZatizeniCerpani,
                PredictedZatizeniCerpani = forecast.PredictedZatizeniCerpani
            });
        }
        return forecasts.Select(x => x.PredictedZatizeniCerpani).ToList();
    }

    public List<float> Sdca(List<Value> trainData, int scope)
    {
        var data = context.Data.LoadFromEnumerable(trainData);
        var pipeline = context.Transforms.CopyColumns("Label", "zatizeniCerpani")
            .Append(context.Transforms.Concatenate("Features", "Year", "Month", "Day", "Hour"))
            .Append(context.Regression.Trainers.Sdca("Label", "Features"));

        var model = pipeline.Fit(data);
        var predictionEngine = context.Model.CreatePredictionEngine<Value, SingleValueForecast>(model);
        var forecasts = new List<SingleValueForecast>();

        var last = trainData.Last();
        var nextDate = last.date;

        for (int i = 0; i < scope; i++)
        {
            nextDate = nextDate.AddHours(1);
            var nextValue = new Value(nextDate, 0);
            var forecast = predictionEngine.Predict(nextValue);
            forecasts.Add(new SingleValueForecast
            {
                Year = nextValue.Year,
                Month = nextValue.Month,
                Day = nextValue.Day,
                Hour = nextValue.Hour,
                zatizeniCerpani = forecast.PredictedZatizeniCerpani,
                PredictedZatizeniCerpani = forecast.PredictedZatizeniCerpani
            });
        }
        return forecasts.Select(x => x.PredictedZatizeniCerpani).ToList();
    }

    public List<float> Gam(List<Value> trainData, int scope)
    {
        var data = context.Data.LoadFromEnumerable(trainData);
        var pipeline = context.Transforms.CopyColumns("Label", "zatizeniCerpani")
            .Append(context.Transforms.Concatenate("Features", "Year", "Month", "Day", "Hour"))
            .Append(context.Regression.Trainers.Gam("Label", "Features"));

        var model = pipeline.Fit(data);
        var predictionEngine = context.Model.CreatePredictionEngine<Value, SingleValueForecast>(model);
        var forecasts = new List<SingleValueForecast>();

        var last = trainData.Last();
        var nextDate = last.date;

        for (int i = 0; i < scope; i++)
        {
            nextDate = nextDate.AddHours(1);
            var nextValue = new Value(nextDate, 0);
            var forecast = predictionEngine.Predict(nextValue);
            forecasts.Add(new SingleValueForecast
            {
                Year = nextValue.Year,
                Month = nextValue.Month,
                Day = nextValue.Day,
                Hour = nextValue.Hour,
                zatizeniCerpani = forecast.PredictedZatizeniCerpani,
                PredictedZatizeniCerpani = forecast.PredictedZatizeniCerpani
            });
        }
        return forecasts.Select(x => x.PredictedZatizeniCerpani).ToList();
    }
}