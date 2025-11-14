
# Goals
1. Learn how to use my GA system `fa-durable-ga` to perform feature-engineering over a dataset.
2. Create a simple (as possible) demonstration repository of how to to use `Rust`, `Burn` and `fa-durable-ga` together to do so.
3. Find some interesting and meaningful relationships, and achieve some good prediction results.
4. Write a LinkedIn and/or a blog post sharing the process, insights and results

## Dataset options
Link:
https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations

Additional resources:
https://github.com/Dalageo/ML-GasSensorDrift

## Approach
1. Be able to load the dataset, training and validation sets
2. Online processing EMA & ZSCORE (lets just try these..?)
3. Configure GA to change the EMA window, alpha and ZSCORE configurations..?
4. Run and evaluate?

## TODO
 - add inference benchmark - run the model against one of the data files for a timeframe, produce CSV with actual and predicted results
 - split by random samples rather than temporally (better generalization)
 - cache preprocessed data
 - make wind direction representable as radians - currently it's just None.
 - add spatial awareness - find coordinates for each station and then compute polar coodinate from the centroid.
 - add one-hot encoding for the station
 - interpolare forward fill values

## How to tell this story
### Goals
  - Bring attention to my skillset, I want to attract a fun job eventually (long term)
  - I want to find like minded, curious people and possible collaborations
  - I want to share own learnings, and get others perspectives on my conclusions and findings. Im genuinly curious if there is a lot of people that do automated feature engineering or if that's somehow not a thing. I have not heard much about it,

### Key narrative
  - Tie together with my previous post about `fx-durable-ga`, make clear this is one experiement using it, as part of my ML-learning journey in Rust
  - The foundational hyphothesis is that humans are poor at knowing which features and parameters to use - that is a job for an algoritm that has the tenacity to stay structured at all times.

### Parts
  - Introduction, background, and goals
  - Making sense of the data
    - walkthrough of what we got, and what challanges we needed to solve
    - encode time
    - encode space
    - create a simple but flexible preprocessing system
  - Setting up the model crate
    - Constraints and decisions - keeping it fast and simple with a feed forward network
    - Refer anything about model architecture, burn internals to documentation, none of that shall be covered
    - CLI tool setup - crating a flexible syntax
    - Playing with the model - got to 4MSE after introducing month
  - Integrating the model crate with `fx-durable-ga`
    - Bring up key aspects, don't go through all of the code
  - Optimization results
    - Human benchmark

### Posts
[x] Publishing announcement `fx-durable-ga`
[ ] Automated feature engineering <-- THIS PROJECT
[ ] Multi machine cooperative optimizations
[ ] Using real time data, live predictions
[ ] Re-optimizing and automatic deploys
[ ] Combining alternative data sources

On testing:
`prediction`, `actual_value` and `current_value` all use the same unit. In this case, its Celcius (temperature), so at timestep 100, the current temperature was -0.4 degree Celcius, and the next actual value (at timestep 101), was -1.1 degree celsius.
```sh
row_number,prediction,actual_value,current_value
100,1.6517326,-1.1,-0.4
101,0.025137715,-1.9,-1.1
```

That gives us a baseline: -1.1 is a 100% correct prediction, and at distance 0.7 degree celcius the prediction is as good as guessing the current value, we call that 0% correct, anything less than 0.7 degree celcius from the target value adds no value.

```sh
prediction  actual_value    current_value,  dist_baseline   dist_prediction
1.65        -1.1            -0.4            0.7             2.75
0.02        -1.9            -1.1            0.8             1.92
```

`dist_baseline = ABS(actual_value - current_value)`
`dist_prediction = ABS(actual_value - prediction)`

This should give us an idea if predictions are are better or worse than just guessing that the next value will be the current value.

We could also from this compute an an accuracy score like:
```sh
accuracy = dist_prediction > dist_baseline ? 0 : dist_baseline / dist_prediction
```

If that value is higher than 0, then we can argue that the prediction provides some value, since it's better than just guessing that things will not change.
