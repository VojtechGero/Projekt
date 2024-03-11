using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Semestral
{
    internal static class Utilities
    {
        internal static List<Value> getData(string filePath)
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

        internal static List<float> errors(List<float> actualData, List<float> forecastData)
        {
            List<float> errors = new List<float>();
            for (int i = 0; i < int.Min(actualData.Count, forecastData.Count); i++)
            {
                errors.Add(actualData[i] - forecastData[i]);
            }
            return errors;
        }

        internal static double MeanAbsoluteError(List<float> error)
        {
            return error.Average(x=>Math.Abs(x));
        }

        internal static double RootMeanSquare(List<float> error)
        {
            return Math.Sqrt(error.Average(x=>x*x));
        }

        internal static void evaluate(List<Value> TestData, List<float> actual)
        {
            List<float> forecast = TestData.Select(s => s.zatizeniCerpani).ToList();
            var metrics = errors(actual, forecast);
            var MAE = MeanAbsoluteError(metrics);
            var RMSE = RootMeanSquare(metrics);
            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine("---------------------");
            Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
            Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
        }

    }
}
