using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Semestral
{
    internal static class Utilities
    {
        

        internal static float[] errors(List<float> actualData, List<float> forecastData)
        {
            float[] errors = new float[int.Min(actualData.Count, forecastData.Count)];
            for (int i = 0; i < int.Min(actualData.Count, forecastData.Count); i++)
            {
                errors[i]=actualData[i] - forecastData[i];
            }
            return errors;
        }

        internal static double MeanAbsoluteError(float[] error)
        {
            return error.Average(x=>Math.Abs(x));
        }

        internal static double RootMeanSquare(float[] error)
        {
            return Math.Sqrt(error.Average(x=>x*x));
        }

        internal static double MeanAbsolutePercentageError(List<float> actualData, List<float> forecastData)
        {
            double sum = 0;
            for (int i = 0; i < int.Min(actualData.Count, forecastData.Count); i++)
            {
                sum += Math.Abs((actualData[i] - forecastData[i]) / actualData[i]) * 100;
            }
            return sum/ int.Min(actualData.Count, forecastData.Count);
        }

        internal static void evaluate(List<Value> TestData, List<float> actual)
        {
            List<float> forecast = TestData.Select(s => s.zatizeniCerpani).ToList();
            var error = errors(actual, forecast);
            var per = MeanAbsolutePercentageError(actual, forecast);
            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine("---------------------\n");
            Console.WriteLine($"Mean Absolute Error: {MeanAbsoluteError(error):F3}");
            Console.WriteLine($"Root Mean Squared Error: {RootMeanSquare(error):F3}");
            Console.WriteLine($"Mean absolute percentage error: {per:F3}");
            Console.WriteLine($"Mean accuracy: {100-per:F3}\n");

        }

    }
}
