baselines_v01:
I choose naive for 10, 22 days and moving average for 66, 132 days
    - naive for 10 and 22 horizon with these distances is better (naive - moving)
                               10:     Score: 8.26965458578169 - 8.56871129175309 = -0.299
                                     Average: 0.30628350317709 - 0.31735967747233 = -0.011 
    
                               22:     Score: 4.245103359239346 - 4.273622990895386 = -0.028
                                     Average: 0.3537586132699455 - 0.35613524924128215 = -0.003

    - moving average for 66 and 132 horizon these distances is better:
                               66:     Score: 1.7945943166470029 - 1.6659032490989485 = 0.129
                                     Average: 0.4486485791617507 - 0.4164758122747371 = 0.032 
    
                               132:    Score: 1.2364947147347025 - 0.8406599853847296 = 0.396
                                     Average: 0.6182473573673513 - 0.4203299926923648 = 0.198

SARIMA, ARIMA:
3 method for interpolation. for some materials, all results are same. for those results that are not same, the order goes to candidate orders and choosen after running cross-validation.
we can run SARIMAX with none values! So decide after cross-validation. 
We are not worried about missing values in this phase, because SARIMAX can handle missed values.

Run with interpolation.
Most of the models have same result.
use shared result, if there is difference based on interpolation.
add model to run SARIMAX directly

Material:
- 1: 
  - interpolation doesn't have any impact.
  - seasonal order have more than 2 unit better AICC --> 
  
- 2: 
  - interpolation doesn't have any impact.
  - seasonal order have no parameter, so one order --> Arima (1, 1, 1)

- 3:
  - no interpolation impact
  - near results, check 2 

Ridge:
    some lags: 1, 5, 10
               1, 2, 5, 10
               1, 2, 3, 4, 5, 10 --> good
               1, 2, 3, 4, 5, 10, rolling10 --> better

Calendar features: day of week --> one hot encoding --> some material
Is_holiday: just some small improvement. in this material day of week was better --> remove

In Ridge, we try 0.0001, 0.001, 0.01, 1 and choose the best one. sometimes differences are not too much but it doesn't add any complexity,
so the less sMAPE is selected.