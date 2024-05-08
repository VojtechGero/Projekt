using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Semestral;

public class SingleValueForecast
{
    public float Year { get; set; }
    public float Month { get; set; }
    public float Day { get; set; }
    public float Hour { get; set; }
    public float zatizeniCerpani { get; set; }
    public DateTime date { get; set; }

    [ColumnName("Score")]
    public float PredictedZatizeniCerpani { get; set; }

    public SingleValueForecast(DateTime date, float zatizeniCerpani)
    {
        this.date = date;
        this.zatizeniCerpani = zatizeniCerpani;
        this.Year = date.Year;
        this.Month = date.Month;
        this.Day = date.Day;
        this.Hour = date.Hour;
    }

    public SingleValueForecast()
    { }

    public Value toValue()
    {
        return new Value(date,zatizeniCerpani);
    }

}
