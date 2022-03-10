# RNN stock price prediction

Small Recurrent Neural Network that predicts stocks


## Disclaimer

This work is based on this [Deep learning course](https://www.udemy.com/course/deeplearning/)

This is for educational purposes only

## Description

This RNN learns data from `dataset_train` which contains 5 years of Google 
stock prices and predicts the next day using X number of days from the past with
`timesteps` (default to 60)

You can use and improve this code freely.

### Improvements

- RNN Optimization (more layers, better hyperparameters, more epochs, etc...)

- Get more and newer data from yahoo finance or other finance sources

### Recommendations

I highly recommend you use [Spyder IDE](https://www.spyder-ide.org/) to run and 
analyze what this RNN is doing, specially on the reshape steps, since those
can be very confusing. With spyder you can run line by line and analize
the variables generated.

![Spyder UI](images/readme_screenshot.png?raw=true "Plot generated from RNN")