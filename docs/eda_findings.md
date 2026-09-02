# EDA Findings (used in preprocessing)

## 1. Input Series

### Data structure and date coverage

**Finding**

- The raw Input dataset contains 1896 daily observations.
- The observation period ranges from 2019-10-24 to 2024-12-31.
- No duplicate dates were detected.
- No calendar dates are missing within the observed period.

### Data quality

**Finding**

- No missing (`NA`) Input values were detected in the raw dataset.
- 248 observations have an Input value of zero.
- 4 observations have negative Input values.
2 occurred on Sundays and two on weekdays (Monday and Tuesday) that one of them is holiday.
- zero values:
  - Zero Input count: 248
  - Weekday: Weekends zeros: 165, Weekday zeros: 83
  - Public holidays: Holiday zeros: 19, Non-holiday zeros: 229
  - Zero-block length distribution:
length
1     119
2      13
3       4
5       1
6       1
7       1
73      1

    start_date   end_date  length
    2023-05-16  2023-07-27     73
    2023-12-07  2023-12-13      7
    2024-10-25  2024-10-30      6



**Decision**

- Negative Input values are considered invalid observations. 
During preprocessing, values below zero are replaced with `NA` (missing values), while their timestamps are important and we don't want to remove.

- Zero values are generally retained as valid observations. 
However, consecutive zero blocks longer than five calendar days are considered abnormal (time periods, doesn't show any valid reason) periods and are treated as missing observations.
For such blocks, the zero values are replaced with `NA` while the corresponding dates are preserved.
The threshold of five days is fixed during model development and is applied consistently to both the historical training data and the 2025 data.

- The resulting missing values are not imputed during common data preparation. 
Missing-value handling is performed later according to the requirements of each forecasting model (because in ARIMA, the sequence of dates is important but in ML algorithms not)

Rules:
1. Input < 0
   → NA
   → timestamp preserved

2. Consecutive Input == 0 block > 5 calendar days
   → zero values in that block become NA
   → timestamps preserved


## 2. Material Fractions

### Data structure and date coverage

**Finding**
- 
- The raw material-fraction dataset contains 1546 observations.
- The observation period ranges from 2020-03-02 to 2024-12-31.
- No duplicate dates were detected.
- 220 calendar dates are absent from the raw dataset within this period.

### Data quality

**Finding**

- 220 calendar dates are absent from the raw material-fraction dataset.
- 160 of the missing dates occur on weekends.
- 60 missing dates occur on weekdays.
- Of these 60 missing weekdays, 11 are public holidays and 49 are non-holiday weekdays.
- No missing values were detected within the 1546 existing observations.
- Every existing row contains values (include 0) for all 13 material fractions.
So observed_mask can be generated for all Materials.
- There is no negtive value is Material fractions.
- Zero values occur frequently and their frequency differs substantially between materials.
- Materials 12 and 13 contain particularly large numbers of zero observations.
- 1,202 of the 1,546 observed dates contain at least one zero-valued material fraction.
- No observed date contains zero values for all 13 materials simultaneously.
- 9 observations contain zero values for 12 of the 13 materials.
- Min, Max value for sum of Material fractions is 1. No negative or 0 values.


**Decision**

- The forecasting time series is aligned to a complete business-day calendar before any model-specific missing-value handling.
- Missing business-day observations are explicitly represented as `NA` rather than removing their timestamps. 
This preserves the correct temporal spacing required for lagged and rolling features.
- We can select this strategy, just use predicted values in last step for next missing values in new train dataset!
  (if there is no actual value for a fold, that fold doesn't participate in evaluation!)
- Treating missing values, should apply after dividing data set to Train-Test.
- Zero-valued material fractions are retained as valid observations. 
Unlike the Input series, no general zero-block rule is applied to individual material fractions.
- No preprocessing rule for negative fraction values is required based on the historical training data. 
- Observed fraction rows are expected to satisfy the following validation constraints:
  - each of the 13 material fractions must lie within the interval [0, 1];
  - all 13 material fractions must be present for an observed row;
  - the sum of the 13 material fractions must equal 1 within numerical tolerance.

## 3. Calendar and External Variables

### 3.1 Weekends
...

### 3.2 Public holidays
...

### 3.3 School holidays
...

## 4. Final Preprocessing Rules
... 

